import json  # used to parse each NDJSON line from the streamed response
import shutil  # used to check if the ollama binary exists on PATH
import subprocess  # used to run ollama commands and start the server
import time  # used to wait after starting ollama before hitting the API
import requests  # used to make HTTP calls to the ollama API

OLLAMA_BASE = "http://localhost:11434"  # base URL for all ollama API endpoints


# ── Step 1: check ollama is installed ────────────────────────────────────────

if shutil.which("ollama") is None:  # look for the ollama binary on PATH
    raise SystemExit("Ollama is not installed. Install it from https://ollama.com")  # hard stop if missing

print("✓ ollama binary found")  # confirm the binary exists


# ── Step 2: ensure ollama server is running ───────────────────────────────────

try:  # attempt a lightweight GET to check if the server is already up
    requests.get(OLLAMA_BASE, timeout=2)  # 2-second timeout so we fail fast
    print("✓ ollama server is running")  # server responded, nothing to do
except requests.exceptions.ConnectionError:  # nothing listening on the port
    print("Ollama server not running — starting it...")  # inform the user
    subprocess.Popen(["ollama", "serve"])  # launch `ollama serve` as a background daemon
    time.sleep(3)  # give the server a few seconds to initialize


# ── Step 3: run `ollama list` and parse available models ─────────────────────

result = subprocess.run(  # run `ollama list` as a subprocess and capture output
    ["ollama", "list"],  # the command to run
    capture_output=True,  # capture stdout and stderr instead of printing
    text=True,  # decode output as a string instead of bytes
)

lines = result.stdout.strip().splitlines()  # split output into individual lines

if len(lines) <= 1:  # only the header row means no models are pulled
    raise SystemExit("No models found. Run `ollama pull llama3.2` to download one.")  # tell user what to do

models = [line.split()[0] for line in lines[1:]]  # first column of each data row is the model name

print(f"✓ models available: {models}")  # show what's on the machine


# ── Step 4: pick the first available model ───────────────────────────────────

model = models[0]  # use the first model returned by `ollama list`

print(f"✓ using model: {model}")  # confirm which model was selected


# ── Step 5: make the generate call ───────────────────────────────────────────

payload = {  # build the request body
    "model": model,  # use the model resolved above
    "prompt": "Why is the sky blue? Answer in one sentence.",  # the prompt to send
    "stream": True,  # get back one complete JSON response, not a streamed sequence
}

t_start = time.perf_counter()  # record time before sending the request

response = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, stream=True)  # stream=True keeps the connection open so we can read line by line
response.raise_for_status()  # raise if the server returned an error status

t_first_token = None  # will hold the time when the first token arrives
token_count = 0  # count tokens to calculate throughput

print()  # blank line before the response starts
for line in response.iter_lines():  # iterate over each NDJSON line as it arrives
    if not line:  # skip empty lines
        continue
    chunk = json.loads(line)  # parse the JSON object from this line
    if t_first_token is None:  # only set this once, on the very first token
        t_first_token = time.perf_counter()  # time to first token (TTFT)
    token = chunk["response"]  # extract the token string from this chunk
    print(token, end="", flush=True)  # print the token immediately without a newline
    token_count += len(token.split())  # rough word count as a proxy for tokens
    if chunk.get("done"):  # ollama sets done=true on the final chunk
        t_end = time.perf_counter()  # record time when generation finished
        break  # stop reading once the response is complete

print()  # newline after the streamed response finishes
print(f"\n── timing ───────────────────────────────")
print(f"  time to first token : {t_first_token - t_start:.2f}s")  # latency before streaming began
print(f"  total response time : {t_end - t_start:.2f}s")  # wall-clock time for the full response
print(f"  words generated     : {token_count}")  # approximate output size
