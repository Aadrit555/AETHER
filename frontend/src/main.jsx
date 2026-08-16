import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "Request failed");
  return data;
}




function App() {
  const [page, setPage] = useState("train");

  return <div className="app"><aside><strong>Aether</strong><button onClick={() => setPage("train")}>Train</button><button onClick={() => setPage("data")}>Data</button><button onClick={() => setPage("test")}>Test</button><button onClick={() => setPage("models")}>Models</button></aside><main>{page === "train" && <Train />}{page === "data" && <Data />}{page === "test" && <Test />}{page === "models" && <Models />}</main></div>
}

function Models() { const [q, setQ] = useState("qwen 0.5b"), [items, setItems] = useState([]), [error, setError] = useState(""); async function search() { try { setItems((await request("/models/search", { method: "POST", body: JSON.stringify({ q, limit: 12 }) })).models) } catch (e) { setError(e.message) } } useEffect(() => { search() }, []); return <Page title="Choose model" subtitle="Search Hugging Face live."><div className="search"><input value={q} onChange={e => setQ(e.target.value)} onKeyDown={e => e.key === 'Enter' && search()} /><button onClick={search}>Search</button></div>{error && <p className="error">{error}</p>}<div className="cards">{items.map(x => <div className="card" key={x.model_id}><b>{x.model_id}</b><span>{(x.downloads || 0).toLocaleString()} downloads</span><small>{x.pipeline_tag || "text generation"}</small></div>)}</div></Page> }

function Data() { const [q, setQ] = useState("instruction dataset"), [items, setItems] = useState([]), [file, setFile] = useState(null), [result, setResult] = useState(null), [error, setError] = useState(""); async function search() { try { setItems((await request("/datasets/search", { method: "POST", body: JSON.stringify({ q, limit: 12 }) })).datasets) } catch (e) { setError(e.message) } } async function upload() { if (!file) return; const form = new FormData(); form.append("file", file); const r = await fetch(`${API}/datasets/upload`, { method: "POST", headers: { Authorization: `Bearer ${localStorage.token}` }, body: form }); const d = await r.json(); if (!r.ok) throw new Error(d.detail); setResult(d) } async function useHub(id) { try { setResult(await request(`/datasets/from-hub?dataset_id=${encodeURIComponent(id)}`, { method: "POST" })) } catch (e) { setError(e.message) } } return <Page title="Choose data" subtitle="Use your own file or find a dataset on Hugging Face."><div className="panel"><input type="file" accept=".json,.jsonl,.csv" onChange={e => setFile(e.target.files[0])} /><button onClick={upload}>Use my file</button></div><div className="search"><input value={q} onChange={e => setQ(e.target.value)} onKeyDown={e => e.key === 'Enter' && search()} /><button onClick={search}>Find datasets</button></div>{error && <p className="error">{error}</p>}{result && <div className="success">Dataset ready: {result.clean_rows ?? result.report?.valid} usable rows. ID: {result.dataset_id}</div>}<div className="cards">{items.map(x => <div className="card" key={x.dataset_id}><b>{x.dataset_id}</b><span>{(x.downloads || 0).toLocaleString()} downloads</span><button className="secondary" onClick={() => useHub(x.dataset_id)}>Use dataset</button></div>)}</div></Page> }

function Train() { const [model, setModel] = useState("Qwen/Qwen2.5-0.5B-Instruct"), [datasetId, setDatasetId] = useState(""), [epochs, setEpochs] = useState(1), [job, setJob] = useState(null), [error, setError] = useState(""); async function start() { try { setError(""); const d = await request("/training/start", { method: "POST", body: JSON.stringify({ model_id: model, dataset_id: datasetId, epochs: Number(epochs) }) }); setJob(d); poll(d.job_id) } catch (e) { setError(e.message) } } async function poll(id) { const timer = setInterval(async () => { try { const d = await request(`/training/${id}`); setJob(d); if (["completed", "failed"].includes(d.status)) clearInterval(timer) } catch (e) { clearInterval(timer); setError(e.message) } }, 1500) } return <Page title="Train model" subtitle="Pick model and dataset. Aether handles training defaults."><label>Model ID<input value={model} onChange={e => setModel(e.target.value)} /></label><label>Dataset ID<input value={datasetId} onChange={e => setDatasetId(e.target.value)} placeholder="Use Data page first" /></label><label>Epochs<input type="number" min="1" max="10" value={epochs} onChange={e => setEpochs(e.target.value)} /></label><button onClick={start}>Start training</button>{error && <p className="error">{error}</p>}{job && <div className="progress"><b>{job.status}</b><div className="bar"><i style={{ width: `${job.progress || 0}%` }} /></div><p>{job.message}</p>{job.result_path && <small>Adapter ready. Job ID: {job.id}</small>}</div>}</Page> }

function Test() { const [model, setModel] = useState("Qwen/Qwen2.5-0.5B-Instruct"), [job, setJob] = useState(""), [prompt, setPrompt] = useState("Explain recursion simply."), [answer, setAnswer] = useState(""), [error, setError] = useState(""); async function run() { try { setError(""); setAnswer((await request("/inference", { method: "POST", body: JSON.stringify({ model_id: model, adapter_job_id: job || null, prompt }) })).answer) } catch (e) { setError(e.message) } } return <Page title="Test model" subtitle="Generate from real model weights."><input value={model} onChange={e => setModel(e.target.value)} /><input value={job} onChange={e => setJob(e.target.value)} placeholder="Completed training job ID (optional)" /><textarea value={prompt} onChange={e => setPrompt(e.target.value)} /><button onClick={run}>Generate</button>{error && <p className="error">{error}</p>}{answer && <div className="answer">{answer}</div>}</Page> }

function Page({ title, subtitle, children }) { return <><header><strong>Aether</strong><button className="link" onClick={() => location.reload()}>Refresh</button></header><section className="hero"><p className="eyebrow">AETHER</p><h1>{title}</h1><p>{subtitle}</p></section><section className="content">{children}</section></> }

createRoot(document.getElementById("root")).render(<App />);
