use crate::{OperationSuccess, SystemRegisterCommand};
use std::{future::Future, pin::Pin};

pub(crate) const N_WORKERS: u64 = 69;
pub(crate) type BoxedCmd = Box<SystemRegisterCommand>;
pub(crate) type Callback =
    Box<dyn FnOnce(OperationSuccess) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;
