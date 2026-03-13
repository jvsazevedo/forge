# F# + Rust ML Framework — Roadmap

> "Alguém tem que dar o primeiro passo."

## Visão Geral

Reescrever a stack de treinamento Stochastic Thinking com uma arquitetura
de duas linguagens que espelha o modelo Python/C++ — mas com linguagens melhores:

```
F#    (pesquisa: arquitetura, forward, training loop, configs, eval)
  |   compila em segundos — muda o tempo todo
  |   P/Invoke / FFI
  v
Rust  (infra: tensor ops, autograd, CUDA, tokenizer, safetensors)
  |   compila uma vez — raramente muda durante experimento
  |   FFI
  v
C / CUDA  (kernels, cuBLAS, cuDNN)
```

**Por que não só TorchSharp?** TorchSharp (binding .NET → libtorch) funcionaria,
mas adiciona uma camada de indirection sobre C++ que já é uma camada sobre CUDA.
Com Rust no meio, ganhamos: ecossistema HF nativo (tokenizers, safetensors, candle),
controle sobre o backend, e a possibilidade de contribuir upstream.

**Por que não só Rust?** Rust é verboso pra experimentação. Borrow checker + compile
times de 30s+ a cada mudança de hiperparâmetro matam o feedback loop de pesquisa.
F# compila em <2s, tem type inference excelente, e pipeline operators tornam
forward passes legíveis como pseudocódigo.

**Princípio:** Cada fase produz algo utilizável e verificável contra a versão Python.

---

## Stack Tecnológica

### Camada Rust — `storch-core`

**Candle** (HuggingFace) como base:

- Tensor ops + autograd + CUDA — maduro, testado, mantido pelo HF
- SafeTensors loading — **nativo** (HF fez os dois, zero-copy)
- Modelos Llama já implementados (referência pra validação)

**HF Tokenizers** — a lib Python é um binding. O original é Rust.

- BPE, WordPiece, Unigram
- `tokenizer.json` loading
- Tokens especiais, padding, truncation

**Exposição:** Rust compila pra shared library (`.so`/`.dll`) com interface C:

```rust
#[no_mangle]
pub extern "C" fn storch_tensor_matmul(a: *const Tensor, b: *const Tensor) -> *mut Tensor
```

### Camada F# — `storch`

Consome a lib Rust via P/Invoke:

```fsharp
[<DllImport("storch_core")>]
extern IntPtr storch_tensor_matmul(IntPtr a, IntPtr b)
```

Wrappers idiomáticos em cima:

```fsharp
let forward input =
    input
    |> embed
    |> runLayers 0 14
    |> stBlock tau
    |> runLayers 15 29
    |> norm
    |> lmHead
```

---

## O Que Muda vs Roadmap Anterior

| Componente | Antes (F# + TorchSharp) | Agora (F# + Rust) |
|---|---|---|
| SafeTensors | Implementar do zero | **grátis** (candle) |
| Tokenizer | Implementar ou bindar | **grátis** (HF tokenizers é Rust) |
| Tensor ops / CUDA | TorchSharp (binding C++) | candle (Rust nativo) |
| Autograd | TorchSharp | candle |
| Transformer base | Implementar em F# | candle tem Llama (referência) |
| Overhead estimate | ~14-22 semanas | **~10-15 semanas** |

---

## Fase 0 — Proof of Concept (1 semana)

**Objetivo:** Provar que o pipeline F# → Rust → CUDA funciona end-to-end.

### 0.1 Setup Rust

- [ ] Criar crate `storch-core` com candle como dependência
- [ ] Implementar MLP de 2 camadas usando candle tensors
- [ ] Forward + backward com candle autograd
- [ ] Compilar como cdylib (shared library)
- [ ] Expor via `extern "C"`: create_tensor, matmul, backward, step

### 0.2 Setup F#

- [ ] Criar projeto F# (.NET 8)
- [ ] P/Invoke pra `storch_core.so`
- [ ] Wrapper `Tensor` type com IDisposable (prevent leaks)
- [ ] Treinar MLP em MNIST chamando Rust por baixo
- [ ] Verificar: CUDA funciona, accuracy >95%

### 0.3 Medir

- [ ] Overhead FFI: tempo de 1000 chamadas F# → Rust vs Rust puro
- [ ] Overhead total: F#/Rust vs Python/PyTorch no mesmo MLP
- [ ] Se overhead FFI >5% por chamada, investigar batching de ops

**Entregável:** F# treina MLP via Rust backend com CUDA.
**Go/No-Go:** Se overhead total >25% vs Python, reavaliar arquitetura.

---

## Fase 1 — Rust Core: Bindings Fundamentais (2 semanas)

**Objetivo:** Expor todas as operações necessárias do candle como C API.

### 1.1 Tensor API

```rust
// Criação
storch_tensor_zeros(shape, dtype, device) -> *mut Tensor
storch_tensor_from_slice(data, shape, dtype) -> *mut Tensor
storch_tensor_load_safetensors(path, key) -> *mut Tensor

// Operações
storch_tensor_matmul(a, b) -> *mut Tensor
storch_tensor_add(a, b) -> *mut Tensor
storch_tensor_mul_scalar(a, s) -> *mut Tensor
storch_tensor_softmax(a, dim) -> *mut Tensor
storch_tensor_rms_norm(a, weight, eps) -> *mut Tensor
storch_tensor_silu(a) -> *mut Tensor
storch_tensor_sigmoid(a) -> *mut Tensor
storch_tensor_softplus(a) -> *mut Tensor
storch_tensor_clamp(a, min, max) -> *mut Tensor
storch_tensor_norm(a, dim) -> *mut Tensor
storch_tensor_reshape(a, shape) -> *mut Tensor
storch_tensor_cat(tensors, dim) -> *mut Tensor
storch_tensor_gather(a, dim, index) -> *mut Tensor

// Autograd
storch_tensor_backward(loss)
storch_tensor_grad(param) -> *mut Tensor
storch_tensor_detach(a) -> *mut Tensor
storch_tensor_no_grad_start()
storch_tensor_no_grad_end()

// Memória
storch_tensor_free(t)
```

### 1.2 Tokenizer API

```rust
storch_tokenizer_load(path) -> *mut Tokenizer
storch_tokenizer_encode(tok, text, add_special) -> *mut EncodedIds
storch_tokenizer_decode(tok, ids) -> *mut c_char
storch_tokenizer_add_special(tok, token) -> u32  // retorna token_id
storch_tokenizer_vocab_size(tok) -> u32
storch_tokenizer_free(tok)
```

### 1.3 SafeTensors API

```rust
storch_safetensors_load(path) -> *mut SafeTensorsFile
storch_safetensors_get_tensor(file, name) -> *mut Tensor
storch_safetensors_list_keys(file) -> *mut StringArray
storch_safetensors_free(file)
```

- [ ] Implementar todas as funções acima
- [ ] Testes unitários em Rust pra cada uma
- [ ] Error handling via código de retorno + storch_last_error()

**Entregável:** `storch-core` shared library com API C estável.

---

## Fase 2 — F# Wrappers Idiomáticos (2 semanas)

**Objetivo:** Camada F# que torna o Rust invisível pro usuário.

### 2.1 Tensor Type

```fsharp
type Tensor =
    val mutable private handle: IntPtr

    member this.MatMul(other: Tensor) = Tensor(Native.matmul(this.handle, other.handle))
    member this.Add(other: Tensor) = Tensor(Native.add(this.handle, other.handle))

    // Operadores
    static member (*)(a: Tensor, b: Tensor) = a.MatMul(b)
    static member (+)(a: Tensor, b: Tensor) = a.Add(b)

    interface IDisposable with
        member this.Dispose() = Native.free(this.handle)
```

### 2.2 Module System

```fsharp
// Abstrações pra camadas — sem herança, sem classes abstratas
type IModule =
    abstract Forward: Tensor -> Tensor
    abstract Parameters: unit -> Tensor list

type Linear = {
    Weight: Tensor
    Bias: Tensor option
}
with interface IModule with
    member this.Forward(x) = x.MatMul(this.Weight) + (this.Bias |> Option.defaultValue Tensor.Zero)
    member this.Parameters() = [ this.Weight ] @ (this.Bias |> Option.toList)

// Composição via pipeline
let sequential (layers: IModule list) (input: Tensor) =
    layers |> List.fold (fun x layer -> layer.Forward(x)) input
```

### 2.3 Configs Type-Safe

```fsharp
type Activation = SiLU | ReLU | GELU | Sigmoid | Softplus
type NormType = RMSNorm of eps:float | LayerNorm of eps:float
type Precision = BFloat16 | Float32
type Device = CPU | CUDA of int

type TrainingConfig = {
    Lr: float
    WeightDecay: float
    WarmupSteps: int
    TotalSteps: int
    Precision: Precision
    Device: Device
}

type Phase = Identity | FullTraining
type Category = Math | Factual | Default | Creative

let tauTarget = function
    | Math     -> 0.015f
    | Factual  -> 0.07f
    | Default  -> 0.09f
    | Creative -> 0.14f
```

### 2.4 Tokenizer Wrapper

```fsharp
type Tokenizer =
    val private handle: IntPtr

    static member Load(path: string) = ...
    member this.Encode(text: string) : int array = ...
    member this.Decode(ids: int array) : string = ...
    member this.AddSpecialToken(token: string) : int = ...
    member this.VocabSize : int = ...
```

### 2.5 SafeTensors Wrapper

```fsharp
module SafeTensors =
    let load (path: string) : Map<string, Tensor> = ...
    let loadKey (path: string) (key: string) : Tensor = ...
    let listKeys (path: string) : string list = ...
```

- [ ] Todos os wrappers acima
- [ ] Testes: Tokenizer produz IDs idênticos ao HF Python
- [ ] Testes: SafeTensors carrega SmolLM2-135M corretamente
- [ ] GC-safe: prevent use-after-free, prevent leaks

**Entregável:** API F# limpa e type-safe sobre o core Rust.

---

## Fase 3 — SmolLM2-135M Inference (2-3 semanas)

**Objetivo:** Forward pass completo do SmolLM2 em F#/Rust.

### 3.1 Componentes (em F#, chamando Rust pra ops pesadas)

```fsharp
// RotaryEmbedding
let rotaryEmbed (q: Tensor) (k: Tensor) (positions: Tensor) : Tensor * Tensor = ...

// GQA: 9 heads, 3 KV heads
let groupedQueryAttention (q: Tensor) (k: Tensor) (v: Tensor) (mask: Tensor) : Tensor = ...

// LlamaDecoderLayer
type LlamaLayer = {
    SelfAttn: GroupedQueryAttention
    MLP: GateMLP  // gate_proj + up_proj → SiLU → down_proj
    InputNorm: RMSNorm
    PostAttnNorm: RMSNorm
}

// Full model
type SmolLM2 = {
    Embedding: Tensor       // (49152, 576)
    Layers: LlamaLayer[]    // 30 layers
    FinalNorm: RMSNorm
    LMHead: Tensor          // (576, 49152)
}
```

### 3.2 Operações Rust que podem ser necessárias adicionar

- [ ] `storch_rotary_embed(q, k, cos, sin)` — se performance importar
- [ ] `storch_flash_attention(q, k, v, mask)` — candle tem implementação
- [ ] `storch_gqa_forward(q, k, v, n_heads, n_kv_heads, mask)` — op composta

### 3.3 Verificação

```
Pra cada camada:
  1. Salvar input/output do Python com torch.save()
  2. Carregar em F# via storch
  3. Rodar forward
  4. max(abs(F# - Python)) < 1e-5
```

- [ ] RoPE: mesmos cos/sin que Python
- [ ] Attention: mesmos scores
- [ ] GQA: KV expand correto
- [ ] Layer-by-layer: outputs idênticos
- [ ] End-to-end: mesmos logits
- [ ] Greedy decoding: mesmos tokens

**Entregável:** `SmolLM2.load("path") |> SmolLM2.forward tokens` funciona.

---

## Fase 4 — Arquitetura Stochastic Thinking (2-3 semanas)

**Objetivo:** Portar ST Block, SC, TauRouter, e model wrappers.

### 4.1 ST Block v3

```fsharp
type STBlockV3Config = {
    HiddenSize: int      // 576
    MLPHidden: int       // 256
    EpsMax: float32      // 0.3
    TauInit: float32     // 0.09
    InitScale: float32   // 0.05
}

type STBlockV3 = {
    MLP1: PerturbationMLP
    MLP2: CompensationMLP
    TransformerLayer: LlamaLayer
    PreNorm: RMSNorm
    PostNorm: RMSNorm
    TauRaw: Tensor  // parameter, softplus(raw) = tau
}

let forwardSTBlock (block: STBlockV3) (h: Tensor) : Tensor =
    let tau = Tensor.softplus block.TauRaw
    let eps1 = block.MLP1 |> forward h |> epsilonBallClip block.Config.EpsMax
    let hPerturbed = block.PreNorm.Forward(h + tau * eps1)
    let hTransformed = block.TransformerLayer |> forward hPerturbed
    let eps2 = block.MLP2 |> forward hTransformed |> epsilonBallClip block.Config.EpsMax
    block.PostNorm.Forward(hTransformed - tau * eps2) + h  // global residual
```

### 4.2 Stochastic Controller + PonderNet

```fsharp
type SCOutput = {
    FinalHidden: Tensor     // (B, T, D) weighted accumulation
    HaltProbs: Tensor       // (B, max_iter)
    EffectiveSteps: Tensor  // (B,)
}

let ponderLoss (haltProbs: Tensor) (lambdaP: float32) : Tensor =
    let maxIter = haltProbs.Shape.[1]
    let geometric =
        [| for n in 0..maxIter-1 -> (1.0f - lambdaP) ** float32 n * lambdaP |]
        |> Tensor.ofArray
    Tensor.klDiv (haltProbs.Log()) geometric

let rec forwardWithSC (block: STBlockV3) (sc: SC) (h: Tensor) (maxIter: int) : SCOutput =
    let mutable hFinal = Tensor.zerosLike h
    let mutable pRunning = Tensor.onesLike h.[.., ..0, ..0]  // (B, 1)
    let haltProbs = ResizeArray()

    for i in 0..maxIter-1 do
        let hOut = forwardSTBlock block h
        let pHalt = sc.Forward(h, hOut, block.MLP1 |> lastEps)
        haltProbs.Add(pHalt)
        hFinal <- hFinal + pRunning * pHalt * hOut
        pRunning <- pRunning * (1.0f - pHalt)
        h <- hOut

    { FinalHidden = hFinal; HaltProbs = Tensor.stack haltProbs; EffectiveSteps = computeSteps haltProbs }
```

### 4.3 ST Block v4 (agentic, single-position)

```fsharp
type LatentGate = {
    Net: Linear * Linear  // D→64→1, init bias=-2.0
}

type TauRouter = {
    Net: Linear * Linear  // D→64→1, softplus output
}

type CrossAttention = {
    ProjQ: Linear; ProjK: Linear; ProjV: Linear; ProjOut: Linear
    NumHeads: int; HeadDim: int
}

type STBlockV4 = {
    MLP1: PerturbationMLP    // transferido do ST0
    MLP2: CompensationMLP    // transferido do ST0
    CrossAttn: CrossAttention
    Gate1: LatentGate
    Gate2: LatentGate
    TauRouter: TauRouter
    PreNorm: RMSNorm
    PostNorm: RMSNorm
}

let forwardSTBlockV4 (block: STBlockV4) (hSI: Tensor) (context: Tensor) : Tensor =
    let g1 = block.Gate1 |> gateForward hSI    // scalar (0,1)
    let g2 = block.Gate2 |> gateForward hSI
    let tau = block.TauRouter |> routerForward hSI  // per-sample tau
    let eps1 = block.MLP1 |> forward hSI |> epsilonBallClip epsMax
    let hPerturbed = block.PreNorm.Forward(hSI + g1 * tau * eps1)
    let hTransformed = block.CrossAttn |> crossAttnForward hPerturbed context
    let eps2 = block.MLP2 |> forward hTransformed |> epsilonBallClip epsMax
    block.PostNorm.Forward(hTransformed - g2 * tau * eps2) + hSI
```

### 4.4 Model Wrappers

```fsharp
type ST0_SmolLM2 = {
    Base: SmolLM2
    STBlock: STBlockV3
    SC: StochasticController
    InsertAfterLayer: int  // 14
}

type SI_SmolLM2_V4 = {
    Base: SmolLM2
    STBlock: STBlockV4
    SC: StochasticController
    SIEmbedding: Tensor    // (576,) standalone parameter
    SITokenId: int
    InsertAfterLayer: int  // 14
}

// Forward condicional — ST block só dispara em posições <SI>
let forwardSI (model: SI_SmolLM2_V4) (inputIds: Tensor) =
    let siPositions = findSIPositions inputIds model.SITokenId
    let h = model.Base.Embedding |> embed inputIds
    let h = h |> runLayers model.Base.Layers 0 14
    let h =
        match siPositions with
        | None -> h  // sem <SI>, passthrough puro = ST0 vanilla
        | Some pos ->
            let hSI = extractAtPositions h pos
            let context = extractCausalContext h pos
            let hSI' = forwardSTBlockV4 model.STBlock hSI context
            replaceAtPositions h pos hSI'
    h |> runLayers model.Base.Layers 15 29 |> model.Base.FinalNorm.Forward |> lmHead
```

### 4.5 Weight Transfer

```fsharp
let loadST0Weights (st0Path: string) (block: STBlockV4) =
    let st0 = SafeTensors.load st0Path
    // Mapeamento direto
    block.MLP1.Load(st0, prefix="perturbation_mlp")
    block.MLP2.Load(st0, prefix="compensation_mlp")
    block.PreNorm.Load(st0, prefix="pre_norm")
    block.PostNorm.Load(st0, prefix="post_norm")
    // Descarta: transformer_layer, tau_raw (V4 usa CrossAttn e TauRouter)
```

- [ ] ST Block v3 + v4
- [ ] Stochastic Controller + PonderNet loss
- [ ] TauRouter + LatentGates
- [ ] CrossAttention
- [ ] Model wrappers (ST0 e V4)
- [ ] Weight transfer ST0 → V4
- [ ] Verificação: carregar checkpoints Python, outputs idênticos

**Entregável:** Arquitetura ST completa em F#.

---

## Fase 5 — Training Framework (2-3 semanas)

**Objetivo:** Training loop completo.

### 5.1 Operações Rust a Adicionar

```rust
// Otimizador
storch_adamw_new(params, lr, betas, weight_decay, eps) -> *mut Optimizer
storch_adamw_step(opt)
storch_adamw_zero_grad(opt)
storch_adamw_set_lr(opt, lr)

// Gradient
storch_clip_grad_norm(params, max_norm) -> f32
storch_tensor_requires_grad(t, bool)

// Mixed precision
storch_autocast_start(dtype)
storch_autocast_end()

// Loss
storch_cross_entropy(logits, targets, ignore_index) -> *mut Tensor
storch_mse_loss(pred, target) -> *mut Tensor
storch_kl_div(log_probs, targets) -> *mut Tensor
```

### 5.2 Scheduler em F# (puro, sem dependência)

```fsharp
let cosineWithWarmup (warmupSteps: int) (totalSteps: int) (baseLr: float) (step: int) : float =
    if step < warmupSteps then
        baseLr * float step / float warmupSteps
    else
        let progress = float (step - warmupSteps) / float (totalSteps - warmupSteps)
        baseLr * 0.5 * (1.0 + cos(Math.PI * progress))
```

### 5.3 Loss Functions

```fsharp
type LossResult =
    | ST0Loss of lm: Tensor * ponder: Tensor
    | AgenticLoss of lm: Tensor * tau: Tensor * ponder: Tensor

let combineLoss (ponderBeta: float32) (tauWeight: float32) = function
    | ST0Loss(lm, ponder) -> lm + ponderBeta * ponder
    | AgenticLoss(lm, tau, ponder) -> lm + tauWeight * tau + ponderBeta * ponder
```

### 5.4 Training Loop

```fsharp
let trainPhase2 (model: ST0_SmolLM2) (config: TrainingConfig) (data: Dataset) =
    let optimizer = AdamW.create (model.TrainableParams()) config.Lr config.WeightDecay
    let schedule = cosineWithWarmup config.WarmupSteps config.TotalSteps config.Lr

    for epoch in 0..config.Epochs-1 do
        for batch in data.Batches(config.BatchSize) do
            optimizer.ZeroGrad()
            let output = model.Forward(batch.InputIds, useSC=true, maxIter=config.MaxIter)
            let loss =
                ST0Loss(
                    crossEntropy output.Logits batch.Labels,
                    ponderLoss output.HaltProbs config.GeometricLambda)
                |> combineLoss config.PonderBeta 0.0f
            loss.Backward()
            clipGradNorm (model.TrainableParams()) config.MaxGradNorm |> ignore
            optimizer.Step()
            optimizer.SetLr(schedule optimizer.StepCount)

        // Held-out eval
        let ppl = evalPerplexity model data.HeldOut
        let hs = evalHellaSwag model data.HellaSwag
        log { Epoch = epoch; PPL = ppl; HellaSwag = hs }

        // Abort check
        if ppl > config.BaselinePPL * 1.10f then abort "PPL degradou"
```

### 5.5 Dataset

```fsharp
type TextSample = { InputIds: int[]; Labels: int[]; AttentionMask: int[] }
type AgenticSample = { InputIds: int[]; Labels: int[]; SIPosition: int; TauTarget: float32; Category: Category }

let loadJsonl (path: string) : JsonValue seq = ...
let tokenizeAndPad (tokenizer: Tokenizer) (maxLen: int) (text: string) : TextSample = ...
```

- [ ] Rust: AdamW, gradient ops, loss functions, autocast
- [ ] F#: Scheduler, training loop, dataset, checkpointing
- [ ] F#: Logging JSON lines (compatível com scripts de análise)
- [ ] F#: Abort criteria

**Entregável:** Treina ST0 do zero em F#/Rust.

---

## Fase 6 — Benchmarks + Eval (1-2 semanas)

- [ ] Perplexity
- [ ] HellaSwag (log-likelihood, 4-way)
- [ ] LAMBADA (word prediction)
- [ ] Agentic LAMBADA (com `<SI>`)
- [ ] Tau routing analysis (per-category, R²)
- [ ] Gate analysis

**Entregável:** Benchmarks idênticos ao Python (tolerância 0.1%).

---

## Fase 7 — Reprodução + Open Source (2-3 semanas)

### 7.1 Reprodução

- [ ] Treinar ST0-SmolLM2 em F#/Rust
- [ ] Bater: PPL 13.44, HellaSwag 0.440, LAMBADA 0.238
- [ ] Treinar Exp 2 (agentic)
- [ ] Comparar todos os benchmarks

### 7.2 Publicação

**Repos:**

```
storch-core  (Rust, crate no crates.io)
  - candle-based tensor ops
  - C API pra FFI
  - tokenizer + safetensors wrappers

storch        (F#, pacote NuGet)
  - Tensor wrappers + module system
  - Transformer building blocks
  - Training framework
  - Benchmark suite
```

- [ ] README com exemplos
- [ ] Testes (Rust: cargo test, F#: Expecto/xUnit)
- [ ] CI/CD (GitHub Actions)
- [ ] Licença: Apache 2.0
- [ ] Publicar crate + NuGet

---

## Timeline

| Fase | Duração | Dependência |
|------|---------|-------------|
| 0 - PoC F#→Rust | 1 sem | nenhuma |
| 1 - Rust core bindings | 2 sem | Fase 0 |
| 2 - F# wrappers | 2 sem | Fase 1 |
| 3 - SmolLM2 inference | 2-3 sem | Fase 2 |
| 4 - ST architecture | 2-3 sem | Fase 3 |
| 5 - Training framework | 2-3 sem | Fase 4 |
| 6 - Benchmarks | 1-2 sem | Fase 3 |
| 7 - Reprodução + publish | 2-3 sem | Fases 5+6 |

**Fases 3 e 6 podem ser parcialmente paralelas** (benchmarks de inference
não dependem de training).

**Total: ~10-15 semanas** (solo dev, dedicação parcial).
Caminho crítico: 0 → 1 → 2 → 3 → 4 → 5 → 7.

---

## Riscos

### R1: Candle não suporta operação X

**Prob:** Baixa-média. Candle é menos completo que PyTorch.
**Mitigação:** Candle permite custom ops em Rust. Pior caso: implementa o kernel.
Fase 0 testa as operações críticas (matmul, softmax, RMSNorm, autograd de MLP).

### R2: FFI overhead

**Prob:** Baixa. Chamadas FFI são ~nanossegundos. Tensor ops são ~microssegundos+.
**Mitigação:** Medir na Fase 0. Se necessário, batch multiple ops numa única chamada FFI.

### R3: Candle autograd incompleto

**Prob:** Média. Candle não tem todas as ops com backward implementado.
**Mitigação:** Verificar na Fase 0 que backward funciona pra: linear, softmax,
cross_entropy, sigmoid, softplus, norm, cat, stack, reshape. Se faltar algo,
implementar backward custom em Rust (candle permite).

### R4: Divergência numérica

**Prob:** Média. Candle pode usar implementação diferente de ops vs PyTorch.
**Mitigação:** Testes camada-por-camada na Fase 3.

### R5: Perda de momentum

**Prob:** Média-alta.
**Mitigação:** Rodar Exp 2 em Python em paralelo. Só migrar pesquisa depois da Fase 4.

---

## O Que a Comunidade Ganha

1. **storch-core**: Crate Rust que expõe candle + tokenizers + safetensors como C API
   — qualquer linguagem com FFI pode usar (Julia, Zig, Nim, etc.)
2. **storch**: Primeiro framework F# sério pra ML research
   — type-safe, composível, pipeline operators, DU pra configs
3. **Prova de conceito**: ML research real (paper com resultados) feita fora do Python
4. **Ponte Rust↔F#**: Pattern reutilizável pra qualquer projeto que quer
   Rust performance com F# ergonomics

---

## Decisões Tomadas

1. **Rust como backend** via candle (não TorchSharp/libtorch)
2. **F# como interface** de pesquisa (não Rust direto)
3. **FFI via C ABI** (extern "C" + P/Invoke)
4. **Dois repos**: storch-core (Rust) + storch (F#)

## Decisões em Aberto

1. **Nome definitivo**: `storch`? outro?
2. **Licença**: Apache 2.0? MIT?
3. **Candle vs Burn**: Fase 0 testa candle. Se autograd for limitado, considerar Burn
4. **Monorepo vs multi-repo**: proposta atual é 2 repos, mas monorepo pode ser mais fácil pro CI
