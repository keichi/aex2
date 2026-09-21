//! Which files a client may open.
//!
//! The rule is enforced even in the trusted environments AEX targets, because
//! directory traversal happens by accident as often as on purpose: a client
//! joining a path itself, a notebook with a stale working directory.

use std::path::{Path, PathBuf};

use crate::error::{Result, ServerError};

/// The directories files may be opened under.
///
/// Roots are resolved once at startup and requested paths are resolved on every
/// open, so a symlink out of a root is caught: both sides of the comparison
/// have their symlinks already followed.
#[derive(Debug, Clone)]
pub struct PathPolicy {
    roots: Vec<PathBuf>,
}

impl PathPolicy {
    /// Resolve the roots. A root that does not exist is a configuration error:
    /// serving nothing is never what the operator meant.
    pub fn new(roots: &[PathBuf]) -> Result<Self> {
        if roots.is_empty() {
            return Err(ServerError::Config(
                "no data roots configured: no file could be opened".to_string(),
            ));
        }
        let mut resolved = Vec::with_capacity(roots.len());
        for root in roots {
            let canonical = std::fs::canonicalize(root)
                .map_err(|e| ServerError::Config(format!("data root {}: {e}", root.display())))?;
            if !canonical.is_dir() {
                return Err(ServerError::Config(format!(
                    "data root {} is not a directory",
                    root.display()
                )));
            }
            resolved.push(canonical);
        }
        Ok(PathPolicy { roots: resolved })
    }

    /// The resolved roots.
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// Resolve a client-supplied path to a file under one of the roots.
    ///
    /// A relative path is tried under each root in order, so a client can say
    /// `ocean/sst.npy` without knowing where the server keeps its data. An
    /// absolute path is taken as given and then checked.
    pub fn resolve(&self, requested: &str) -> Result<PathBuf> {
        let resolved = self.locate(requested)?;
        if !resolved.is_file() {
            return Err(ServerError::BadRequest(format!(
                "{requested} is not a regular file"
            )));
        }
        Ok(resolved)
    }

    /// Resolve a client-supplied path to a store under one of the roots.
    ///
    /// A store is a directory a backend reads many files out of, rather than
    /// one file. Which of those files may be opened is the backend's to
    /// enforce; this only settles which directory it may work in.
    pub fn resolve_store(&self, requested: &str) -> Result<PathBuf> {
        let resolved = self.locate(requested)?;
        if !resolved.is_dir() {
            return Err(ServerError::BadRequest(format!(
                "{requested} is not a directory"
            )));
        }
        Ok(resolved)
    }

    /// The path a request names, once it is known to be inside a root.
    fn locate(&self, requested: &str) -> Result<PathBuf> {
        if requested.is_empty() {
            return Err(ServerError::BadRequest("empty path".to_string()));
        }
        let requested = Path::new(requested);

        let candidates: Vec<PathBuf> = if requested.is_absolute() {
            vec![requested.to_path_buf()]
        } else {
            self.roots.iter().map(|root| root.join(requested)).collect()
        };

        let mut last_error = None;
        for candidate in &candidates {
            match std::fs::canonicalize(candidate) {
                Ok(resolved) => {
                    if !self.roots.iter().any(|root| resolved.starts_with(root)) {
                        // Report the path the client asked for: the resolved one
                        // would disclose where symlinks point.
                        return Err(ServerError::PathNotAllowed(format!(
                            "{} is outside the configured data roots",
                            requested.display()
                        )));
                    }
                    return Ok(resolved);
                }
                Err(e) => last_error = Some(e),
            }
        }

        // Every candidate failed to resolve; the last reason is as good as any.
        let reason = last_error.expect("at least one candidate is tried");
        Err(ServerError::Core(aex_core::AexError::Io(
            std::io::Error::new(reason.kind(), format!("{}: {reason}", requested.display())),
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    /// A root holding `data.npy` and a subdirectory `sub/nested.npy`.
    fn root_with_files() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("data.npy"), b"x").unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        fs::write(dir.path().join("sub/nested.npy"), b"x").unwrap();
        dir
    }

    #[test]
    fn resolves_relative_and_absolute_paths_inside_a_root() {
        let dir = root_with_files();
        let policy = PathPolicy::new(&[dir.path().to_path_buf()]).expect("policy");
        let root = fs::canonicalize(dir.path()).unwrap();

        assert_eq!(policy.resolve("data.npy").unwrap(), root.join("data.npy"));
        assert_eq!(
            policy.resolve("sub/nested.npy").unwrap(),
            root.join("sub/nested.npy")
        );
        let absolute = root.join("data.npy");
        assert_eq!(
            policy.resolve(absolute.to_str().unwrap()).unwrap(),
            absolute
        );
    }

    #[test]
    fn a_relative_path_is_tried_under_every_root() {
        let first = root_with_files();
        let second = tempfile::tempdir().unwrap();
        fs::write(second.path().join("other.npy"), b"x").unwrap();
        let policy = PathPolicy::new(&[first.path().to_path_buf(), second.path().to_path_buf()])
            .expect("policy");

        assert!(policy.resolve("data.npy").is_ok());
        assert_eq!(
            policy.resolve("other.npy").unwrap(),
            fs::canonicalize(second.path()).unwrap().join("other.npy")
        );
    }

    #[test]
    fn rejects_a_path_outside_the_roots() {
        let dir = root_with_files();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret.npy"), b"x").unwrap();
        let policy = PathPolicy::new(&[dir.path().to_path_buf()]).expect("policy");

        let absolute = fs::canonicalize(outside.path()).unwrap().join("secret.npy");
        let err = policy.resolve(absolute.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, ServerError::PathNotAllowed(_)), "{err}");
    }

    #[test]
    fn traversal_out_of_a_root_is_rejected() {
        let dir = root_with_files();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret.npy"), b"x").unwrap();
        let policy = PathPolicy::new(&[dir.path().to_path_buf()]).expect("policy");

        let escape = format!(
            "../{}/secret.npy",
            fs::canonicalize(outside.path())
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
        );
        // The temp dirs are siblings, so this really does resolve to the file.
        let err = policy.resolve(&escape).unwrap_err();
        assert!(matches!(err, ServerError::PathNotAllowed(_)), "{err}");
    }

    #[test]
    fn a_symlink_out_of_a_root_is_rejected() {
        let dir = root_with_files();
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("secret.npy");
        fs::write(&target, b"x").unwrap();
        std::os::unix::fs::symlink(&target, dir.path().join("link.npy")).unwrap();
        let policy = PathPolicy::new(&[dir.path().to_path_buf()]).expect("policy");

        // Comparing before resolving symlinks would have let this through.
        let err = policy.resolve("link.npy").unwrap_err();
        assert!(matches!(err, ServerError::PathNotAllowed(_)), "{err}");
    }

    #[test]
    fn reports_a_missing_file_as_not_found() {
        let dir = root_with_files();
        let policy = PathPolicy::new(&[dir.path().to_path_buf()]).expect("policy");
        let err = policy.resolve("absent.npy").unwrap_err();
        assert_eq!(err.class(), aex_core::ErrorClass::Request);
        assert!(err.to_string().contains("absent.npy"), "{err}");
    }

    #[test]
    fn rejects_a_directory_and_an_empty_path() {
        let dir = root_with_files();
        let policy = PathPolicy::new(&[dir.path().to_path_buf()]).expect("policy");
        assert!(matches!(
            policy.resolve("sub"),
            Err(ServerError::BadRequest(_))
        ));
        assert!(matches!(
            policy.resolve(""),
            Err(ServerError::BadRequest(_))
        ));
    }

    #[test]
    fn a_store_resolves_only_when_it_is_a_directory() {
        let dir = root_with_files();
        let policy = PathPolicy::new(&[dir.path().to_path_buf()]).expect("policy");
        let root = fs::canonicalize(dir.path()).unwrap();

        assert_eq!(policy.resolve_store("sub").unwrap(), root.join("sub"));
        // A file is no more a store than a directory is a file.
        assert!(matches!(
            policy.resolve_store("data.npy"),
            Err(ServerError::BadRequest(_))
        ));
        assert!(matches!(
            policy.resolve_store(""),
            Err(ServerError::BadRequest(_))
        ));
    }

    #[test]
    fn a_store_outside_the_roots_is_rejected() {
        let dir = root_with_files();
        let outside = tempfile::tempdir().unwrap();
        fs::create_dir(outside.path().join("secret.zarr")).unwrap();
        let policy = PathPolicy::new(&[dir.path().to_path_buf()]).expect("policy");

        let absolute = fs::canonicalize(outside.path())
            .unwrap()
            .join("secret.zarr");
        let err = policy
            .resolve_store(absolute.to_str().unwrap())
            .unwrap_err();
        assert!(matches!(err, ServerError::PathNotAllowed(_)), "{err}");

        // A link is followed before the roots are compared, here as for a file.
        std::os::unix::fs::symlink(&absolute, dir.path().join("link.zarr")).unwrap();
        let err = policy.resolve_store("link.zarr").unwrap_err();
        assert!(matches!(err, ServerError::PathNotAllowed(_)), "{err}");
    }

    #[test]
    fn a_missing_root_is_a_configuration_error() {
        let err = PathPolicy::new(&[PathBuf::from("/nonexistent/aex2/data")]).unwrap_err();
        assert!(matches!(err, ServerError::Config(_)), "{err}");
        assert!(PathPolicy::new(&[]).is_err());

        let dir = root_with_files();
        let err = PathPolicy::new(&[dir.path().join("data.npy")]).unwrap_err();
        assert!(matches!(err, ServerError::Config(_)), "{err}");
    }
}
