use rustats;
use trackable::error::{ErrorKind as TrackableErrorKind, ErrorKindExt};
use trackable::error::{Failure, TrackableError};

/// This crate specific `Error` type.
#[derive(Debug, Clone, TrackableError)]
pub struct Error(TrackableError<ErrorKind>);
impl From<Failure> for Error {
    fn from(f: Failure) -> Self {
        ErrorKind::Other.takes_over(f).into()
    }
}
impl From<std::io::Error> for Error {
    fn from(f: std::io::Error) -> Self {
        ErrorKind::IoError.cause(f).into()
    }
}
impl From<rustats::Error> for Error {
    fn from(f: rustats::Error) -> Self {
        let kind = match f.kind() {
            rustats::ErrorKind::InvalidInput => ErrorKind::InvalidInput,
            rustats::ErrorKind::Bug => ErrorKind::Bug,
            rustats::ErrorKind::IoError => ErrorKind::IoError,
            rustats::ErrorKind::Other => ErrorKind::Other,
        };
        kind.takes_over(f).into()
    }
}

/// Possible error kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// Invalid input was given.
    InvalidInput,

    /// Unknown observation was given.
    UnknownObservation,

    /// I/O error.
    IoError,

    /// Implementation bug.
    Bug,

    /// Other error.
    Other,
}
impl TrackableErrorKind for ErrorKind {}
