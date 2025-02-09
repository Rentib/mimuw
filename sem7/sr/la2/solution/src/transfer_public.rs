use crate::{
    ClientCommandHeader, ClientRegisterCommand, ClientRegisterCommandContent, RegisterCommand,
    SectorVec, SystemCommandHeader, SystemRegisterCommand, SystemRegisterCommandContent,
    MAGIC_NUMBER,
};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::io::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use uuid::Uuid;

fn verify_client_hmac(msg: &[u8], hmac_tag: &[u8], hmac_client_key: &[u8; 32]) -> bool {
    let mut mac = Hmac::<Sha256>::new_from_slice(hmac_client_key).unwrap();
    mac.update(msg);
    mac.verify_slice(hmac_tag).is_ok()
}

fn verify_system_hmac(msg: &[u8], hmac_tag: &[u8], hmac_system_key: &[u8; 64]) -> bool {
    let mut mac = Hmac::<Sha256>::new_from_slice(hmac_system_key).unwrap();
    mac.update(msg);
    mac.verify_slice(hmac_tag).is_ok()
}

async fn read_u8(data: &mut (dyn AsyncRead + Send + Unpin)) -> Result<u8, Error> {
    let mut buf = [0u8; 1];
    data.read_exact(&mut buf).await?;
    Ok(buf[0])
}

async fn read_u64(data: &mut (dyn AsyncRead + Send + Unpin), bytes: usize) -> Result<u64, Error> {
    let mut buf = [0u8; 8];
    data.read_exact(&mut buf[..bytes]).await?;
    Ok(u64::from_be_bytes(buf))
}

async fn read_4096_bytes(data: &mut (dyn AsyncRead + Send + Unpin)) -> Result<Vec<u8>, Error> {
    let mut buf = [0u8; 4096];
    data.read_exact(&mut buf).await?;
    Ok(buf.to_vec())
}

async fn read_uuid(data: &mut (dyn AsyncRead + Send + Unpin)) -> Result<u128, Error> {
    let mut buf = [0u8; 16];
    data.read_exact(&mut buf).await?;
    Ok(u128::from_be_bytes(buf))
}

async fn read_hmac_tag(data: &mut (dyn AsyncRead + Send + Unpin)) -> Result<[u8; 32], Error> {
    let mut hmac_tag = [0u8; 32];
    data.read_exact(&mut hmac_tag).await?;
    Ok(hmac_tag)
}

pub async fn deserialize_read(
    first_row: &[u8; 4],
    data: &mut (dyn AsyncRead + Send + Unpin),
    _hmac_system_key: &[u8; 64],
    hmac_client_key: &[u8; 32],
) -> Result<(RegisterCommand, bool), Error> {
    let request_identifier = read_u64(data, 8).await?;
    let sector_idx = read_u64(data, 8).await?;
    let hmac_tag = read_hmac_tag(data).await?;

    let msg = {
        let mut msg = Vec::new();
        msg.append(&mut MAGIC_NUMBER.to_vec());
        msg.append(&mut first_row.to_vec());
        msg.append(&mut request_identifier.to_be_bytes().to_vec());
        msg.append(&mut sector_idx.to_be_bytes().to_vec());
        msg
    };

    let cmd = RegisterCommand::Client(ClientRegisterCommand {
        header: ClientCommandHeader {
            request_identifier,
            sector_idx,
        },
        content: ClientRegisterCommandContent::Read,
    });

    Ok((cmd, verify_client_hmac(&msg, &hmac_tag, hmac_client_key)))
}

pub async fn deserialize_write(
    first_row: &[u8; 4],
    data: &mut (dyn AsyncRead + Send + Unpin),
    _hmac_system_key: &[u8; 64],
    hmac_client_key: &[u8; 32],
) -> Result<(RegisterCommand, bool), Error> {
    let request_identifier = read_u64(data, 8).await?;
    let sector_idx = read_u64(data, 8).await?;
    let content = read_4096_bytes(data).await?;
    let hmac_tag = read_hmac_tag(data).await?;

    let msg = {
        let mut msg = Vec::new();
        msg.append(&mut MAGIC_NUMBER.to_vec());
        msg.append(&mut first_row.to_vec());
        msg.append(&mut request_identifier.to_be_bytes().to_vec());
        msg.append(&mut sector_idx.to_be_bytes().to_vec());
        msg.append(&mut content.clone());
        msg
    };

    let cmd = RegisterCommand::Client(ClientRegisterCommand {
        header: ClientCommandHeader {
            request_identifier,
            sector_idx,
        },
        content: ClientRegisterCommandContent::Write {
            data: SectorVec(content),
        },
    });

    Ok((cmd, verify_client_hmac(&msg, &hmac_tag, hmac_client_key)))
}

pub async fn deserialize_read_proc(
    first_row: &[u8; 4],
    data: &mut (dyn AsyncRead + Send + Unpin),
    hmac_system_key: &[u8; 64],
    _hmac_client_key: &[u8; 32],
) -> Result<(RegisterCommand, bool), Error> {
    let process_identifier = first_row[2];
    let uuid = read_uuid(data).await?;
    let sector_idx = read_u64(data, 8).await?;
    let hmac_tag = read_hmac_tag(data).await?;

    let msg = {
        let mut msg = Vec::new();
        msg.append(&mut MAGIC_NUMBER.to_vec());
        msg.append(&mut first_row.to_vec());
        msg.append(&mut uuid.to_be_bytes().to_vec());
        msg.append(&mut sector_idx.to_be_bytes().to_vec());
        msg
    };

    let cmd = RegisterCommand::System(SystemRegisterCommand {
        header: SystemCommandHeader {
            process_identifier,
            msg_ident: Uuid::from_u128(uuid),
            sector_idx,
        },
        content: SystemRegisterCommandContent::ReadProc,
    });

    Ok((cmd, verify_system_hmac(&msg, &hmac_tag, hmac_system_key)))
}

pub async fn deserialize_value(
    first_row: &[u8; 4],
    data: &mut (dyn AsyncRead + Send + Unpin),
    hmac_system_key: &[u8; 64],
    _hmac_client_key: &[u8; 32],
) -> Result<(RegisterCommand, bool), Error> {
    let process_identifier = first_row[2];
    let uuid = read_uuid(data).await?;
    let sector_idx = read_u64(data, 8).await?;
    let timestamp = read_u64(data, 8).await?;
    let padding = {
        let mut padding = [0u8; 7];
        data.read_exact(&mut padding).await?;
        padding
    };
    let write_rank = read_u8(data).await?;
    let sector_data = read_4096_bytes(data).await?;
    let hmac_tag = read_hmac_tag(data).await?;

    let msg = {
        let mut msg = Vec::new();
        msg.append(&mut MAGIC_NUMBER.to_vec());
        msg.append(&mut first_row.to_vec());
        msg.append(&mut uuid.to_be_bytes().to_vec());
        msg.append(&mut sector_idx.to_be_bytes().to_vec());
        msg.append(&mut timestamp.to_be_bytes().to_vec());
        msg.append(&mut padding.to_vec());
        msg.append(&mut [write_rank].to_vec());
        msg.append(&mut sector_data.clone());
        msg
    };

    let cmd = RegisterCommand::System(SystemRegisterCommand {
        header: SystemCommandHeader {
            process_identifier,
            msg_ident: Uuid::from_u128(uuid),
            sector_idx,
        },
        content: SystemRegisterCommandContent::Value {
            timestamp,
            write_rank,
            sector_data: SectorVec(sector_data),
        },
    });

    Ok((cmd, verify_system_hmac(&msg, &hmac_tag, hmac_system_key)))
}

pub async fn deserialize_write_proc(
    first_row: &[u8; 4],
    data: &mut (dyn AsyncRead + Send + Unpin),
    hmac_system_key: &[u8; 64],
    _hmac_client_key: &[u8; 32],
) -> Result<(RegisterCommand, bool), Error> {
    let process_identifier = first_row[2];
    let uuid = read_uuid(data).await?;
    let sector_idx = read_u64(data, 8).await?;
    let timestamp = read_u64(data, 8).await?;
    let padding = {
        let mut padding = [0u8; 7];
        data.read_exact(&mut padding).await?;
        padding
    };
    let write_rank = read_u8(data).await?;
    let sector_data = read_4096_bytes(data).await?;
    let hmac_tag = read_hmac_tag(data).await?;

    let msg = {
        let mut msg = Vec::new();
        msg.append(&mut MAGIC_NUMBER.to_vec());
        msg.append(&mut first_row.to_vec());
        msg.append(&mut uuid.to_be_bytes().to_vec());
        msg.append(&mut sector_idx.to_be_bytes().to_vec());
        msg.append(&mut timestamp.to_be_bytes().to_vec());
        msg.append(&mut padding.to_vec());
        msg.append(&mut [write_rank].to_vec());
        msg.append(&mut sector_data.clone());
        msg
    };

    let cmd = RegisterCommand::System(SystemRegisterCommand {
        header: SystemCommandHeader {
            process_identifier,
            msg_ident: Uuid::from_u128(uuid),
            sector_idx,
        },
        content: SystemRegisterCommandContent::WriteProc {
            timestamp,
            write_rank,
            data_to_write: SectorVec(sector_data),
        },
    });

    Ok((cmd, verify_system_hmac(&msg, &hmac_tag, hmac_system_key)))
}

pub async fn deserialize_ack(
    first_row: &[u8; 4],
    data: &mut (dyn AsyncRead + Send + Unpin),
    hmac_system_key: &[u8; 64],
    _hmac_client_key: &[u8; 32],
) -> Result<(RegisterCommand, bool), Error> {
    let process_identifier = first_row[2];
    let uuid = read_uuid(data).await?;
    let sector_idx = read_u64(data, 8).await?;
    let hmac_tag = read_hmac_tag(data).await?;

    let msg = {
        let mut msg = Vec::new();
        msg.append(&mut MAGIC_NUMBER.to_vec());
        msg.append(&mut first_row.to_vec());
        msg.append(&mut uuid.to_be_bytes().to_vec());
        msg.append(&mut sector_idx.to_be_bytes().to_vec());
        msg
    };

    let cmd = RegisterCommand::System(SystemRegisterCommand {
        header: SystemCommandHeader {
            process_identifier,
            msg_ident: Uuid::from_u128(uuid),
            sector_idx,
        },
        content: SystemRegisterCommandContent::Ack,
    });

    Ok((cmd, verify_system_hmac(&msg, &hmac_tag, hmac_system_key)))
}

pub async fn deserialize_register_command(
    data: &mut (dyn AsyncRead + Send + Unpin),
    hmac_system_key: &[u8; 64],
    hmac_client_key: &[u8; 32],
) -> Result<(RegisterCommand, bool), Error> {
    let mut queue = Vec::new();
    let mut byte = [0u8; 1];
    let mut buf = [0u8; 4];

    loop {
        loop {
            data.read_exact(&mut byte).await?;
            queue.push(byte[0]);
            if queue.len() > MAGIC_NUMBER.len() {
                queue.remove(0);
            }
            if queue == MAGIC_NUMBER {
                break;
            }
        }

        data.read_exact(&mut buf).await?;
        let msg_type = buf[3];

        let res = match msg_type {
            0x01 => Some(deserialize_read(&buf, data, hmac_system_key, hmac_client_key).await),
            0x02 => Some(deserialize_write(&buf, data, hmac_system_key, hmac_client_key).await),
            0x03 => Some(deserialize_read_proc(&buf, data, hmac_system_key, hmac_client_key).await),
            0x04 => Some(deserialize_value(&buf, data, hmac_system_key, hmac_client_key).await),
            0x05 => {
                Some(deserialize_write_proc(&buf, data, hmac_system_key, hmac_client_key).await)
            }
            0x06 => Some(deserialize_ack(&buf, data, hmac_system_key, hmac_client_key).await),
            _ => None,
        };

        if let Some(res) = res {
            return res;
        }
    }
}

async fn serialize_client_register_command(
    cmd: &ClientRegisterCommand,
    writer: &mut (dyn AsyncWrite + Send + Unpin),
    hmac_key: &[u8],
) -> Result<(), Error> {
    let request_identifier = cmd.header.request_identifier;
    let sector_idx = cmd.header.sector_idx;
    let (msg_type, mut content) = match cmd.clone().content {
        ClientRegisterCommandContent::Read => (0x01, Vec::with_capacity(0)),
        ClientRegisterCommandContent::Write { data } => (0x02, data.0),
    };

    let mut buf = Vec::new();

    buf.append(&mut MAGIC_NUMBER.to_vec());
    buf.append(&mut [0x00; 3].to_vec());
    buf.append(&mut [msg_type].to_vec());
    buf.append(&mut request_identifier.to_be_bytes().to_vec());
    buf.append(&mut sector_idx.to_be_bytes().to_vec());
    buf.append(&mut content);

    let mut mac = Hmac::<Sha256>::new_from_slice(hmac_key).unwrap();
    mac.update(&buf);
    let hmac_tag = mac.finalize().into_bytes();

    buf.append(&mut hmac_tag.to_vec());

    writer.write_all(&buf).await?;

    Ok(())
}

async fn serialize_system_register_command(
    cmd: &SystemRegisterCommand,
    writer: &mut (dyn AsyncWrite + Send + Unpin),
    hmac_key: &[u8],
) -> Result<(), Error> {
    let process_identifier = cmd.header.process_identifier;
    let msg_ident = cmd.header.msg_ident;
    let sector_idx = cmd.header.sector_idx;
    let (msg_type, mut content) = match cmd.clone().content {
        SystemRegisterCommandContent::ReadProc => (0x03, Vec::with_capacity(0)),
        SystemRegisterCommandContent::Value {
            timestamp,
            write_rank,
            sector_data,
        } => {
            let mut content = Vec::new();
            let mut sector_data = sector_data.0;
            content.append(&mut timestamp.to_be_bytes().to_vec());
            content.append(&mut [0x00; 7].to_vec());
            content.append(&mut [write_rank].to_vec());
            content.append(&mut sector_data);
            (0x04, content)
        }
        SystemRegisterCommandContent::WriteProc {
            timestamp,
            write_rank,
            data_to_write,
        } => {
            let mut content = Vec::new();
            let mut sector_data = data_to_write.0;
            content.append(&mut timestamp.to_be_bytes().to_vec());
            content.append(&mut [0x00; 7].to_vec());
            content.append(&mut [write_rank].to_vec());
            content.append(&mut sector_data);
            (0x05, content)
        }
        SystemRegisterCommandContent::Ack => (0x06, Vec::with_capacity(0)),
    };

    let mut buf = Vec::new();

    buf.append(&mut MAGIC_NUMBER.to_vec());
    buf.append(&mut [0x00, 0x00].to_vec());
    buf.append(&mut process_identifier.to_be_bytes().to_vec());
    buf.append(&mut [msg_type].to_vec());
    buf.append(&mut msg_ident.as_bytes().to_vec());
    buf.append(&mut sector_idx.to_be_bytes().to_vec());
    buf.append(&mut content);

    let mut mac = Hmac::<Sha256>::new_from_slice(hmac_key).unwrap();
    mac.update(&buf);
    let hmac_tag = mac.finalize().into_bytes();

    buf.append(&mut hmac_tag.to_vec());

    writer.write_all(&buf).await?;

    Ok(())
}

pub async fn serialize_register_command(
    cmd: &RegisterCommand,
    writer: &mut (dyn AsyncWrite + Send + Unpin),
    hmac_key: &[u8],
) -> Result<(), Error> {
    match cmd {
        RegisterCommand::Client(cmd) => {
            serialize_client_register_command(cmd, writer, hmac_key).await
        }
        RegisterCommand::System(cmd) => {
            serialize_system_register_command(cmd, writer, hmac_key).await
        }
    }
}
