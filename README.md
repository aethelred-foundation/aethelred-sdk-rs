<h1 align="center">aethelred-sdk-rs</h1>

<p align="center">
  <strong>Official Rust SDK for the Aethelred blockchain</strong>
</p>

<p align="center">
  <a href="https://github.com/aethelred-foundation/aethelred-sdk-rs/actions/workflows/repo-security-baseline.yml"><img src="https://img.shields.io/github/actions/workflow/status/aethelred-foundation/aethelred-sdk-rs/repo-security-baseline.yml?branch=main&style=flat-square&label=Security" alt="Security"></a>
  <a href="https://crates.io/crates/aethelred"><img src="https://img.shields.io/crates/v/aethelred?style=flat-square&logo=rust" alt="crates.io"></a>
  <a href="https://docs.rs/aethelred"><img src="https://img.shields.io/docsrs/aethelred?style=flat-square" alt="docs.rs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" alt="License"></a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.85+-DEA584?style=flat-square&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/PQC-Kyber+Dilithium-purple?style=flat-square" alt="PQC">
  <img src="https://img.shields.io/badge/async-Tokio-blue?style=flat-square" alt="Tokio">
  <a href="https://docs.aethelred.io/sdk"><img src="https://img.shields.io/badge/docs-SDK-orange?style=flat-square" alt="Docs"></a>
</p>

---

## Install

```toml
# Cargo.toml
[dependencies]
aethelred = "0.1"
tokio = { version = "1", features = ["full"] }
```

## Quick Start

```rust
use aethelred::{AethelredClient, Wallet, pouw::SubmitJobRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Connect to testnet
    let client = AethelredClient::connect("https://rpc.testnet.aethelred.io").await?;

    // Load wallet
    let wallet = Wallet::from_mnemonic("your twelve word mnemonic...")?;

    // Submit an AI compute job
    let job = client.pouw().submit_job(SubmitJobRequest {
        model_hash: hex::decode("abc123...")?,
        input_data: b"{\"prompt\":\"Hello AI\"}".to_vec(),
        verification_type: "hybrid".into(),
        priority: "standard".into(),
        signer: wallet.clone(),
    }).await?;

    println!("Job submitted: {}", job.job_id);

    // Wait for and verify the seal
    let seal = client.seal().await_seal_by_job(&job.job_id).await?;
    println!("Output hash: {}", hex::encode(&seal.output_hash));
    println!("Agreement: {}/{}", seal.agreement_power, seal.total_power);

    Ok(())
}
```

## Features

```toml
[dependencies]
aethelred = { version = "0.1", features = [
    "pouw",      # PoUW module client
    "seal",      # Digital Seal queries
    "verify",    # ZK proof utilities
    "bridge",    # Ethereum bridge
    "tls",       # TLS support
    "wasm",      # WASM target support
]}
```

Full API docs: [docs.rs/aethelred](https://docs.rs/aethelred)

---

## Development

```bash
cargo build
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```
