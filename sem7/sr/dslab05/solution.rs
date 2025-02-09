use std::io::{Read, Write};
use rustls::pki_types::ServerName;
use std::sync::Arc;
use rustls::{ClientConnection, RootCertStore, ServerConnection, StreamOwned};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use rustls::pki_types::pem::PemObject;
// You can add here other imports from std or crates listed in Cargo.toml.

type HmacSha256 = Hmac<Sha256>;

pub struct SecureClient<L: Read + Write> {
    // Add here any fields you need.
    stream: StreamOwned<ClientConnection, L>,
    hmac_key: Vec<u8>,
}

pub struct SecureServer<L: Read + Write> {
    // Add here any fields you need.
    stream: StreamOwned<ServerConnection, L>,
    hmac_key: Vec<u8>,
}

impl<L: Read + Write> SecureClient<L> {
    /// Creates a new instance of SecureClient.
    ///
    /// SecureClient communicates with SecureServer via `link`.
    /// The messages include a HMAC tag calculated using `hmac_key`.
    /// A certificate of SecureServer is signed by `root_cert`.
    /// We are connecting with `server_hostname`.
    pub fn new(
        link: L,
        hmac_key: &[u8],
        root_cert: &str,
        server_hostname: ServerName<'static>,
    ) -> Self {
        let mut root_store = RootCertStore::empty();
        root_store.add_parsable_certificates(rustls::pki_types::CertificateDer::from_pem_slice(
            root_cert.as_bytes(),
        ));
        let client_config = rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();
        let connection = ClientConnection::new(Arc::new(client_config), server_hostname)
            .expect("TLS connection failed");
        let stream = StreamOwned::new(connection, link);
        SecureClient {
            stream,
            hmac_key: hmac_key.to_vec(),
        }
    }

    /// Sends the data to the server. The sent message follows the
    /// format specified in the description of the assignment.
    pub fn send_msg(&mut self, data: Vec<u8>) {
        let length = (data.len() as u32).to_be_bytes();
        
        let mut mac = HmacSha256::new_from_slice(&self.hmac_key).expect("Invalid HMAC key");
        mac.update(&data);
        let hmac_tag = mac.finalize().into_bytes();
        
        let mut message = Vec::new();
        message.extend_from_slice(&length);
        message.extend_from_slice(&data);
        message.extend_from_slice(&hmac_tag);

        self.stream.write_all(&message).expect("Failed to send message");
    }
}

impl<L: Read + Write> SecureServer<L> {
    /// Creates a new instance of SecureServer.
    ///
    /// SecureServer receives messages from SecureClients via `link`.
    /// HMAC tags of the messages are verified against `hmac_key`.
    /// The private key of the SecureServer's certificate is `server_private_key`,
    /// and the full certificate chain is `server_full_chain`.
    pub fn new(
        link: L,
        hmac_key: &[u8],
        server_private_key: &str,
        server_full_chain: &str,
    ) -> Self {
        let certs = rustls::pki_types::CertificateDer::pem_slice_iter(server_full_chain.as_bytes())
            .flatten()
            .collect();
        let private_key =
            rustls::pki_types::PrivateKeyDer::from_pem_slice(server_private_key.as_bytes())
                .expect("Failed to load private key");
        let server_config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, private_key)
            .expect("Failed to create server config");
        let connection = ServerConnection::new(Arc::new(server_config))
            .expect("TLS connection failed");
        let stream = StreamOwned::new(connection, link);
        SecureServer {
            stream,
            hmac_key: hmac_key.to_vec(),
        }
    }

    /// Receives the next incoming message and returns the message's content
    /// (i.e., without the message size and without the HMAC tag) if the
    /// message's HMAC tag is correct. Otherwise, returns `SecureServerError`.
    pub fn recv_message(&mut self) -> Result<Vec<u8>, SecureServerError> {
        let mut length_bytes = [0u8; 4];
        self.stream.read_exact(&mut length_bytes).expect("Failed to read message length");
        let length = u32::from_be_bytes(length_bytes) as usize;
        let mut data = vec![0u8; length];
        self.stream.read_exact(&mut data).expect("Failed to read message content");
        let mut hmac_tag = vec![0u8; 32];
        self.stream.read_exact(&mut hmac_tag).expect("Failed to read HMAC tag");
        let mut mac = HmacSha256::new_from_slice(&self.hmac_key).expect("Invalid HMAC key");
        mac.update(&data);
        if mac.verify_slice(&hmac_tag).is_err() {
            return Err(SecureServerError::InvalidHmac);
        }
        Ok(data)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum SecureServerError {
    /// The HMAC tag of a message is invalid.
    InvalidHmac,
}

// You can add any private types, structs, consts, functions, methods, etc., you need.
// Helper functions to load certificates and keys
