# Infrastructure & API Setup

### 1. Start the infrastructure
```bash
docker compose up -d
````

This will launch:

* **PostgreSQL** (listening on port `5433`)
* **MinIO** (S3 API on port `9000`, web console on port `9001`)

---

### 2. Create the S3 bucket in MinIO

1. Open the MinIO console: [http://localhost:9001](http://localhost:9001)
2. Log in with:

   * **Username:** `symbx`
   * **Password:** `symbx12345`
3. Create a new bucket named **`symbx`**

---

### 3. Run the API service

```bash
make api
```

This will start the **SymbX FastAPI server** on [http://localhost:8000](http://localhost:8000).

---

### 4. Quick demo (in a separate terminal)

```bash
make curl-demo
```

This will:

* Check the health endpoint
* Bootstrap the rules in the database
* Register a sample program `(*3.0; +2.0; *2.0)`
* Execute the program with input `A=1.5` and return the prediction
