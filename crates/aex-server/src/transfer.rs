//! Transfer plans: what the control plane promised and the data plane serves.
//!
//! This registry is the only thing the two planes share. `PrepareSelection`
//! writes an entry, and a connection thread reads it on every `FETCH`. Nothing
//! holds a lock while transferring: a thread takes its `Arc` out of the map and
//! lets the map go.
//!
//! There is no RPC to release a plan. Releasing on the critical path would cost
//! a round trip, and off it would only free memory slightly sooner; an entry is
//! small — an `Arc` to the dataset and the resolved layout — and the count is
//! capped per session. So plans go three ways: they expire, they are evicted
//! when a session has too many, or the session ends.
//!
//! The TTL is pushed out on every `FETCH` rather than fixed at issue. A fixed
//! deadline would make a transfer that legitimately takes longer than the TTL —
//! a large selection over a slow link, or off a cold cache — fail every single
//! time. Extending it means a plan expires only when the client really did go
//! quiet, which is what the setting is supposed to mean.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use aex_core::{ArrayDataset, SelectionLayout};
use aex_wire::Ticket;

use crate::config::ServerConfig;
use crate::error::{Result, ServerError};
use crate::session::{random_bytes, SessionId};

/// One prepared transfer.
pub struct TransferEntry {
    request_id: u32,
    session: SessionId,
    ticket: Ticket,
    dataset: Arc<dyn ArrayDataset>,
    layout: SelectionLayout,
    /// Milliseconds since the registry's epoch, at the last `FETCH`.
    last_used_ms: AtomicU64,
}

impl std::fmt::Debug for TransferEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The ticket is a capability, so it is redacted rather than printed.
        f.debug_struct("TransferEntry")
            .field("request_id", &self.request_id)
            .field("total_bytes", &self.layout.total_bytes)
            .finish()
    }
}

impl TransferEntry {
    /// Identifies this transfer on the data plane.
    pub fn request_id(&self) -> u32 {
        self.request_id
    }

    /// Which session may fetch it.
    pub fn session(&self) -> &SessionId {
        &self.session
    }

    /// The capability a `FETCH` has to present. Never logged.
    pub fn ticket(&self) -> &Ticket {
        &self.ticket
    }

    pub fn dataset(&self) -> &Arc<dyn ArrayDataset> {
        &self.dataset
    }

    pub fn layout(&self) -> &SelectionLayout {
        &self.layout
    }

    fn touch(&self, now_ms: u64) {
        self.last_used_ms.store(now_ms, Ordering::Relaxed);
    }

    fn last_used_ms(&self) -> u64 {
        self.last_used_ms.load(Ordering::Relaxed)
    }

    fn is_expired(&self, now_ms: u64, ttl_ms: u64) -> bool {
        now_ms.saturating_sub(self.last_used_ms()) > ttl_ms
    }
}

/// Every live transfer plan.
#[derive(Debug)]
pub struct TransferRegistry {
    // ponytail: one lock for every plan; it is taken once per RPC or FETCH
    // and held only for a map operation. Shard it if that ever shows up.
    plans: Mutex<Plans>,
    next_id: AtomicU32,
    config: Arc<ServerConfig>,
    /// Plans are timed against a monotonic clock, so a wall-clock jump cannot
    /// expire them all at once.
    epoch: Instant,
}

/// The plans, and the index into them by session. One lock over both keeps
/// them consistent.
#[derive(Debug, Default)]
struct Plans {
    entries: HashMap<u32, Arc<TransferEntry>>,
    /// Which plans belong to which session, for the per-session cap and for
    /// dropping them all when the session goes.
    by_session: HashMap<SessionId, Vec<u32>>,
}

impl TransferRegistry {
    pub fn new(config: Arc<ServerConfig>) -> Self {
        TransferRegistry {
            plans: Mutex::default(),
            // From 1: a request_id of 0 means "about the connection itself" on
            // the data plane, so no transfer may claim it.
            next_id: AtomicU32::new(1),
            config,
            epoch: Instant::now(),
        }
    }

    fn plans(&self) -> MutexGuard<'_, Plans> {
        self.plans.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    fn ttl_ms(&self) -> u64 {
        self.config.limits.transfer_ttl_sec.saturating_mul(1000)
    }

    /// Register a resolved selection and mint its ticket.
    pub fn insert(
        &self,
        session: SessionId,
        dataset: Arc<dyn ArrayDataset>,
        layout: SelectionLayout,
    ) -> Result<Arc<TransferEntry>> {
        self.insert_at(session, dataset, layout, self.now_ms())
    }

    fn insert_at(
        &self,
        session: SessionId,
        dataset: Arc<dyn ArrayDataset>,
        layout: SelectionLayout,
        now_ms: u64,
    ) -> Result<Arc<TransferEntry>> {
        let entry = Arc::new(TransferEntry {
            request_id: self.next_request_id(),
            session,
            ticket: random_bytes()?,
            dataset,
            layout,
            last_used_ms: AtomicU64::new(now_ms),
        });

        let mut plans = self.plans();
        let Plans {
            entries,
            by_session,
        } = &mut *plans;
        let ids = by_session.entry(session).or_default();
        // Plans swept for age leave their ids behind; drop those before
        // deciding the session is at its limit.
        ids.retain(|id| entries.contains_key(id));
        while ids.len() >= self.config.limits.max_transfers_per_session as usize {
            let Some(position) = least_recently_used(entries, ids) else {
                break;
            };
            let evicted = ids.swap_remove(position);
            entries.remove(&evicted);
            tracing::debug!(request_id = evicted, "evicted the oldest transfer plan");
        }
        ids.push(entry.request_id);
        entries.insert(entry.request_id, entry.clone());
        Ok(entry)
    }

    /// Resolve a `FETCH` to the plan it names.
    ///
    /// The ticket and the session are both checked. Without the ticket, a
    /// request_id is a small integer and any other session on this server could
    /// read someone else's transfer by trying a few; this is the only access
    /// control the data plane has.
    pub fn fetch(
        &self,
        request_id: u32,
        ticket: &Ticket,
        session: &SessionId,
    ) -> Result<Arc<TransferEntry>> {
        let now_ms = self.now_ms();
        let entry = self
            .plans()
            .entries
            .get(&request_id)
            .cloned()
            .ok_or_else(|| {
                ServerError::NoSuchPlan(format!(
                    "transfer {request_id} does not exist, or expired after {} seconds of \
                     inactivity",
                    self.config.limits.transfer_ttl_sec
                ))
            })?;

        if entry.is_expired(now_ms, self.ttl_ms()) {
            self.remove(request_id);
            return Err(ServerError::NoSuchPlan(format!(
                "transfer {request_id} expired after {} seconds of inactivity",
                self.config.limits.transfer_ttl_sec
            )));
        }

        // One message for both checks: telling the two apart would say whether
        // a guessed request_id existed.
        if !constant_time_eq(&entry.ticket, ticket) || entry.session != *session {
            return Err(ServerError::Auth(format!(
                "the ticket presented for transfer {request_id} is not the one it was issued"
            )));
        }

        entry.touch(now_ms);
        Ok(entry)
    }

    /// Note that a transfer is still going, so that it does not expire under a
    /// client that is doing exactly what it was told to.
    pub fn touch(&self, entry: &TransferEntry) {
        entry.touch(self.now_ms());
    }

    fn remove(&self, request_id: u32) {
        let mut plans = self.plans();
        if let Some(entry) = plans.entries.remove(&request_id) {
            if let Some(ids) = plans.by_session.get_mut(&entry.session) {
                ids.retain(|id| *id != request_id);
            }
        }
    }

    /// Drop every plan of a session that has ended.
    pub fn remove_session(&self, session: &SessionId) -> usize {
        let mut plans = self.plans();
        let Some(ids) = plans.by_session.remove(session) else {
            return 0;
        };
        for id in &ids {
            plans.entries.remove(id);
        }
        ids.len()
    }

    /// Drop plans nobody has fetched within the TTL, and plans whose session is
    /// gone.
    ///
    /// A session that timed out rather than disconnecting takes its plans with
    /// it here, which is why this needs to be told which sessions are still
    /// live.
    pub fn sweep(&self, is_live: impl Fn(&SessionId) -> bool) -> usize {
        self.sweep_at(self.now_ms(), is_live)
    }

    fn sweep_at(&self, now_ms: u64, is_live: impl Fn(&SessionId) -> bool) -> usize {
        let ttl_ms = self.ttl_ms();
        let mut plans = self.plans();
        let Plans {
            entries,
            by_session,
        } = &mut *plans;
        let before = entries.len();
        entries.retain(|_, entry| !entry.is_expired(now_ms, ttl_ms) && is_live(&entry.session));

        // Sessions whose plans have all gone leave an empty list behind.
        by_session.retain(|session, ids| {
            ids.retain(|id| entries.contains_key(id));
            !ids.is_empty() && is_live(session)
        });
        before - entries.len()
    }

    pub fn len(&self) -> usize {
        self.plans().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plans().entries.is_empty()
    }

    /// The next identifier, never 0.
    ///
    /// The counter wraps after four billion transfers. A wrapped id can only
    /// collide with one of the few plans alive at that moment, and the ticket
    /// check would refuse the fetch rather than serve the wrong bytes.
    fn next_request_id(&self) -> u32 {
        loop {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            if id != 0 {
                return id;
            }
        }
    }
}

/// Where in `ids` the plan nobody has touched for longest is.
fn least_recently_used(entries: &HashMap<u32, Arc<TransferEntry>>, ids: &[u32]) -> Option<usize> {
    ids.iter()
        .enumerate()
        .filter_map(|(position, id)| Some((position, entries.get(id)?.last_used_ms())))
        .min_by_key(|(_, last_used)| *last_used)
        .map(|(position, _)| position)
}

/// Compare two tickets without leaking where they differ.
///
/// The bytes are random and the attacker is across a network, so this is
/// belt-and-braces; it costs nothing at sixteen bytes.
fn constant_time_eq(a: &Ticket, b: &Ticket) -> bool {
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

#[cfg(test)]
mod tests {
    use aex_core::{DType, QualitySpec};

    use super::*;

    /// A dataset of `len` bytes that reads back its own offsets.
    struct FakeDataset {
        shape: Vec<u64>,
    }

    impl ArrayDataset for FakeDataset {
        fn dtype(&self) -> DType {
            DType::Uint8
        }
        fn shape(&self) -> &[u64] {
            &self.shape
        }
        fn read_range(
            &self,
            _layout: &SelectionLayout,
            offset: u64,
            dst: &mut [u8],
        ) -> aex_core::Result<()> {
            for (i, byte) in dst.iter_mut().enumerate() {
                *byte = (offset + i as u64) as u8;
            }
            Ok(())
        }
    }

    fn dataset(len: u64) -> Arc<dyn ArrayDataset> {
        Arc::new(FakeDataset { shape: vec![len] })
    }

    fn layout(len: u64) -> SelectionLayout {
        SelectionLayout::resolve(&[len], DType::Uint8, &[], &QualitySpec::exact()).expect("layout")
    }

    fn registry_with(adjust: impl FnOnce(&mut ServerConfig)) -> TransferRegistry {
        let mut config = ServerConfig::default();
        adjust(&mut config);
        TransferRegistry::new(Arc::new(config))
    }

    fn registry() -> TransferRegistry {
        registry_with(|_| {})
    }

    #[test]
    fn a_plan_is_fetched_with_its_ticket() {
        let registry = registry();
        let session = [1u8; 16];
        let entry = registry
            .insert(session, dataset(1024), layout(1024))
            .expect("insert");

        assert_ne!(entry.request_id(), 0);
        assert_eq!(entry.layout().total_bytes, 1024);
        assert_eq!(registry.len(), 1);

        let fetched = registry
            .fetch(entry.request_id(), entry.ticket(), &session)
            .expect("fetch");
        assert_eq!(fetched.request_id(), entry.request_id());
    }

    #[test]
    fn request_ids_and_tickets_are_unique() {
        let registry = registry();
        let first = registry.insert([1; 16], dataset(8), layout(8)).unwrap();
        let second = registry.insert([1; 16], dataset(8), layout(8)).unwrap();
        assert_ne!(first.request_id(), second.request_id());
        assert_ne!(first.ticket(), second.ticket());
    }

    #[test]
    fn another_session_cannot_fetch_a_plan_even_with_the_right_id() {
        // Without the ticket check, request_ids are small integers and a few
        // guesses would read someone else's transfer.
        let registry = registry();
        let entry = registry.insert([1; 16], dataset(8), layout(8)).unwrap();

        let err = registry
            .fetch(entry.request_id(), entry.ticket(), &[2; 16])
            .unwrap_err();
        assert!(matches!(err, ServerError::Auth(_)), "{err}");

        // Nor with the right session and the wrong ticket.
        let err = registry
            .fetch(entry.request_id(), &[0; 16], &[1; 16])
            .unwrap_err();
        assert!(matches!(err, ServerError::Auth(_)), "{err}");
        // A near miss is still a miss.
        let mut almost = *entry.ticket();
        almost[15] ^= 1;
        assert!(registry
            .fetch(entry.request_id(), &almost, &[1; 16])
            .is_err());
    }

    #[test]
    fn an_unknown_plan_is_reported_as_a_plan_error() {
        // The client re-prepares on this class, and on no other.
        let registry = registry();
        let err = registry.fetch(4242, &[0; 16], &[1; 16]).unwrap_err();
        assert!(matches!(err, ServerError::NoSuchPlan(_)), "{err}");
        assert_eq!(err.class(), aex_core::ErrorClass::Plan);
    }

    #[test]
    fn fetching_pushes_the_deadline_out() {
        // A transfer that takes longer than the TTL must not fail halfway
        // through just for taking its time.
        let registry = registry();
        let ttl = registry.ttl_ms();
        let session = [1u8; 16];
        let entry = registry
            .insert_at(session, dataset(8), layout(8), 0)
            .expect("insert");

        for step in 1..5 {
            let now = step * ttl;
            assert!(!entry.is_expired(now, ttl), "still being fetched");
            entry.touch(now);
        }
        assert!(entry.is_expired(5 * ttl + 1, ttl), "the client went quiet");
    }

    #[test]
    fn an_expired_plan_is_dropped_rather_than_kept_around() {
        let registry = registry();
        let entry = registry
            .insert_at([1; 16], dataset(8), layout(8), 0)
            .unwrap();
        assert_eq!(registry.sweep_at(registry.ttl_ms(), |_| true), 0);
        assert_eq!(registry.sweep_at(registry.ttl_ms() + 1, |_| true), 1);
        assert!(registry.is_empty());
        // And its id no longer resolves.
        assert!(registry
            .fetch(entry.request_id(), entry.ticket(), &[1; 16])
            .is_err());
    }

    #[test]
    fn the_sweep_drops_the_plans_of_a_session_that_timed_out() {
        let registry = registry();
        let live = [1u8; 16];
        let gone = [2u8; 16];
        registry.insert(live, dataset(8), layout(8)).unwrap();
        registry.insert(gone, dataset(8), layout(8)).unwrap();

        assert_eq!(registry.sweep(|session| *session == live), 1);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn a_disconnect_takes_the_sessions_plans_with_it() {
        let registry = registry();
        let session = [1u8; 16];
        for _ in 0..3 {
            registry.insert(session, dataset(8), layout(8)).unwrap();
        }
        registry.insert([2; 16], dataset(8), layout(8)).unwrap();

        assert_eq!(registry.remove_session(&session), 3);
        assert_eq!(registry.len(), 1);
        // Removing it again is not an error; a session ends only once.
        assert_eq!(registry.remove_session(&session), 0);
    }

    #[test]
    fn a_session_at_its_limit_loses_its_oldest_plan() {
        let registry = registry_with(|config| config.limits.max_transfers_per_session = 3);
        let session = [1u8; 16];

        let mut entries = Vec::new();
        for step in 0..3 {
            entries.push(
                registry
                    .insert_at(session, dataset(8), layout(8), step)
                    .unwrap(),
            );
        }
        assert_eq!(registry.len(), 3);

        // Using the oldest one makes it the newest, so the next one out is the
        // one that really has been idle longest.
        registry
            .fetch(entries[0].request_id(), entries[0].ticket(), &session)
            .expect("fetch");
        entries[0].touch(10);

        let fresh = registry
            .insert_at(session, dataset(8), layout(8), 11)
            .unwrap();
        assert_eq!(registry.len(), 3);
        assert!(registry
            .fetch(entries[1].request_id(), entries[1].ticket(), &session)
            .is_err());
        for kept in [&entries[0], &entries[2], &fresh] {
            registry
                .fetch(kept.request_id(), kept.ticket(), &session)
                .expect("kept");
        }
    }

    #[test]
    fn the_limit_is_per_session() {
        let registry = registry_with(|config| config.limits.max_transfers_per_session = 2);
        for session in 0..4u8 {
            for _ in 0..2 {
                registry
                    .insert([session; 16], dataset(8), layout(8))
                    .unwrap();
            }
        }
        assert_eq!(registry.len(), 8);
    }

    #[test]
    fn a_ticket_comparison_does_not_stop_at_the_first_difference() {
        let ticket = [7u8; 16];
        assert!(constant_time_eq(&ticket, &ticket));
        for byte in 0..16 {
            let mut other = ticket;
            other[byte] ^= 0x80;
            assert!(!constant_time_eq(&ticket, &other), "byte {byte}");
        }
    }
}
