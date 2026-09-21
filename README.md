# AGMT --- Elastic Recurrent Translation Architecture

**AGMT** is an experimental elastic recurrent architecture for machine
translation.

This GitHub repository is the **canonical architecture and design
documentation for AGMT**. It describes the model architecture, recurrent
computation mechanism, translation-value routing, memory paths, and the
intended training architecture.

AGMT itself is **an architecture**, not a specific English ↔ Vietnamese
model.

> **Demo model:** An experimental English ↔ Vietnamese \~20M
> implementation of AGMT is available on Hugging Face:
>
> **https://huggingface.co/Phitran21/agmt-en-vi-20m**

------------------------------------------------------------------------

## Core idea

Most translation systems execute a predetermined amount of encoder
computation for every input.

AGMT instead allows the translation representation to be refined
recurrently and asks:

> **Does another refinement step still have useful translation value?**

Conceptually:

``` text
easy input
→ current representation is sufficient
→ HALT

input that can still benefit
→ REFINE
→ evaluate again

difficult input
→ may receive additional recurrent computation

no useful future improvement
→ HALT
```

The goal is not:

``` text
use the deepest recurrence possible
```

and not simply:

``` text
minimize computation
```

The goal is:

``` text
use additional latent computation
only while it remains useful for translation
```

------------------------------------------------------------------------

# Architecture Overview

``` text
SOURCE
  │
  ▼
Shared Byte-Safe Tokenizer
  │
  ▼
Shared Embedding
  │
  ├──────────────────────► Lexical Highway
  │
  ▼
Prelude Encoder
  │
  ▼
H_base
  │
  ▼
Shared Recurrent Refiner ◄──────────────┐
  │                                     │
  ├────► Translation-Value Router       │
  │          ├─ HALT                    │
  │          └─ REFINE ─────────────────┘
  │
  ▼
H_refined
  │
  ├──────────── H_base
  ├──────────── E_lex
  ▼
Stable Memory Bridge
  │
  ▼
Coda Adapter
  │
  ▼
Shallow Autoregressive Decoder
  │
  ▼
TRANSLATION
```

AGMT does **not** claim that every individual component is new.

The architecture combines established neural machine translation and
Transformer concepts with an experimental composition centered around:

-   shared recurrent refinement;
-   elastic effective computation depth;
-   translation-value routing;
-   lexical preservation;
-   a stable pre-recurrence contextual shortcut;
-   a stable decoder memory interface;
-   frozen-oracle router training.

------------------------------------------------------------------------

# Shared Recurrent Refiner

After the Prelude Encoder produces `H_base`, AGMT can repeatedly apply
the same recurrent refinement transformation.

``` text
H0
 │
 ▼
Refiner
 │
 ▼
H1
 │
 ▼
Refiner
 │
 ▼
H2
 │
 ▼
...
```

The refiner parameters are shared.

Conceptually:

``` text
H_(r+1) = Refiner(H_r, StepEncoding(r))
```

This separates:

``` text
stored parameters
```

from:

``` text
effective computation depth
```

Increasing `r` does not create another independent parameter layer. It
executes the shared refiner again on the current hidden representation.

Therefore:

``` text
larger r ≠ automatically better translation
```

A recurrent step may:

``` text
improve the representation
preserve it
or degrade it
```

For example, a possible depth trajectory is:

``` text
r0  incorrect
r1  incorrect
r2  correct
r3  correct
r4  correct
r5  degraded
```

The purpose of adaptive routing is therefore not to find the largest
`r`, but to stop when additional refinement no longer has sufficient
expected translation value.

------------------------------------------------------------------------

# Refinement Step Signal

Because the same refiner parameters are reused, the block receives
information about the current recurrent stage.

Conceptually:

``` text
H_r + StepEncoding(r)
        │
        ▼
 Shared Refiner
        │
        ▼
     H_(r+1)
```

This lets a shared transformation behave differently depending on the
current refinement stage without requiring separate parameters for every
depth.

------------------------------------------------------------------------

# Translation-Value Router

The router is not intended to answer:

``` text
Is this sentence long?
Is this sentence difficult?
Did the hidden state change a lot?
```

Its intended question is:

> **If refinement continues, is there still meaningful translation
> improvement available?**

Conceptually:

``` text
remaining_translation_value > compute_price
→ REFINE

remaining_translation_value <= compute_price
→ HALT
```

At inference time the router is **source-only**.

Reference translations may be used to construct supervision during
training, but they are not available to the routing decision during
inference.

This distinction is important:

``` text
high translation loss
≠
additional recurrence will help
```

An input can be difficult while additional refinement provides almost no
benefit. Such a sample should not automatically receive maximum compute.

------------------------------------------------------------------------

# Router Representation

Routing is intended to use several complementary signals rather than
sentence length or simple mean pooling alone.

Conceptually:

``` text
Global Summary
+
Hard-Token Summary
+
State-Change Summary
+
Direction
+
Refinement Step
+
Weak Length Signal
```

A small number of unresolved tokens should not disappear inside a
sentence-level average.

State change is useful as a feature, but it is not itself the routing
objective.

------------------------------------------------------------------------

# Lexical Highway

The shared embedding branches into a lexical preservation path:

``` text
E
│
├── contextual path
│
└── Lexical Highway → E_lex
```

The purpose of `E_lex` is to preserve source-surface evidence that
should not be forced to survive every recurrent transformation.

Examples include:

``` text
names
numbers
technical identifiers
symbols
short lexical units
rare forms
```

The Lexical Highway is not an independent translator.

It is a shortcut representation.

------------------------------------------------------------------------

# Prelude Encoder and H_base

Before recurrence, the Prelude Encoder produces:

``` text
H_base
```

`H_base` is the contextual foundation of the recurrent process.

It also remains available later as a stable contextual shortcut.

This means the recurrent refiner does not need to perfectly preserve
every piece of early contextual information through every recurrent
step.

------------------------------------------------------------------------

# Stable Memory Bridge

The decoder does not receive only a raw recurrent endpoint.

AGMT preserves three streams:

``` text
H_refined
H_base
E_lex
```

Conceptually:

``` text
H_bridge
=
H_refined
+
g_base · P_base(H_base)
+
g_lex · P_lex(E_lex)
```

Then:

``` text
H_mem = CodaAdapter(H_bridge)
```

`H_refined` remains the primary contextual anchor.

`H_base` provides early contextual information.

`E_lex` provides lexical evidence.

The bridge is intended to reduce information loss across recurrence and
provide a more stable decoder interface across different effective
depths.

------------------------------------------------------------------------

# Coda Adapter

A lightweight adapter follows the Memory Bridge.

Its purpose is to normalize the memory interface presented to the
decoder and reduce distribution differences caused by different
recurrent endpoints.

The Coda Adapter is not intended to become another independent semantic
model.

------------------------------------------------------------------------

# Decoder

AGMT uses a relatively shallow autoregressive decoder.

Its main responsibilities are:

``` text
target-language generation
target syntax
target history
cross-attention to encoder memory
```

The encoder/refiner side carries most of the architecture's adaptive
contextual computation.

------------------------------------------------------------------------

# Translation Is the Final Authority

AGMT is not intended to become:

``` text
small LLM
+
translator
```

Its design principle is:

``` text
translation quality
>
semantic elegance
>
compute saving
```

This means:

-   semantic representations matter when they improve translation;
-   recurrent computation matters when it improves translation;
-   saving compute matters only when translation quality is preserved.

Understanding is a resource for translation, not an independent final
objective.

------------------------------------------------------------------------

# Training Architecture

The intended high-level training process separates translator learning
from final routing control.

``` text
STATIC DATA SANITATION
        │
        ▼
TRAIN TRANSLATOR
        │
        ▼
ESTABLISH MULTI-DEPTH BEHAVIOR
        │
        ▼
FREEZE TRANSLATOR
        │
        ▼
PROFILE VALID RECURRENT DEPTHS
        │
        ▼
BUILD ε-ORACLE / FUTURE-VALUE TARGETS
        │
        ▼
TRAIN SOURCE-ONLY ROUTER
        │
        ▼
CALIBRATE HALTING POLICY
        │
        ▼
EVALUATE
```

The translator is trained before the final router.

Once the translator is stable, it is frozen before final router
supervision is generated.

Otherwise:

``` text
translator changes
→ recurrence behavior changes
→ oracle targets move
→ router learns a moving target
```

Freezing turns final router learning into a more stable control problem.

------------------------------------------------------------------------

# Recurrence Oracle

With a frozen translator, translation quality can be profiled at every
valid recurrent depth:

``` text
L0
L1
L2
...
LR
```

The oracle should not blindly choose:

``` text
deepest recurrence
```

or even blindly choose:

``` text
absolute minimum observed loss
```

Instead, the intended ε-oracle selects the **shallowest recurrent depth
that is already sufficiently close to the best useful translation state
for that sample**.

Conceptually:

``` text
r_oracle
=
smallest r

such that

L_r <= L_best + ε
```

The tolerance exists because tiny differences between recurrent
endpoints may be noise rather than meaningful translation improvements.

------------------------------------------------------------------------

# Remaining Translation Value

The router is intended to learn:

``` text
remaining useful translation value
```

rather than:

``` text
absolute translation loss
```

This distinction prevents a high-loss sample from automatically being
interpreted as a sample that needs more recurrence.

The relevant question is:

``` text
Can future recurrent computation improve this translation enough to matter?
```

not:

``` text
Is the current translation bad?
```

------------------------------------------------------------------------

# Conservative Halting

A false HALT can prevent useful computation from occurring.

For that reason, routing near an uncertain decision boundary should
favor additional refinement when translation quality is the priority.

Different inference policies may later trade computation against quality
without requiring separately trained translation models.

------------------------------------------------------------------------

# Detailed Architecture Specification

**This `README.md` is intentionally a shortened introduction.**

The complete architecture and training specification is maintained in:

## [`readme-agmt.md`](./readme-agmt.md)

That document describes AGMT in substantially more detail, including:

-   tokenizer and shared embedding design;
-   Lexical Highway;
-   Prelude Encoder;
-   recurrent stability;
-   step representations;
-   Translation-Value Router;
-   global and hard-token summaries;
-   state-change features;
-   weak length features;
-   Stable Memory Bridge;
-   Coda Adapter;
-   bidirectional translation;
-   inference policies;
-   force-deep fallback;
-   multi-depth translator training;
-   deep-first vs variable-depth training;
-   frozen translator profiling;
-   recurrence oracle;
-   noise-aware best-future targets;
-   ε-oracle;
-   robust router supervision;
-   catastrophic-loss masking;
-   router feature detachment;
-   semantic challenge evaluation;
-   optional semantic consistency;
-   selective distillation;
-   translation-priority gradient arbitration;
-   long-context considerations;
-   position architecture;
-   truncation policy;
-   technical and named-entity robustness;
-   optional curriculum and data-quality experiments;
-   architecture and training rules.

> If this README leaves an architectural detail unclear, **read
> `readme-agmt.md` before drawing conclusions about AGMT**.\
> `readme-agmt.md` is the detailed specification; this file is the
> entry-level overview.

------------------------------------------------------------------------

# Demo Model

A small experimental implementation exists to demonstrate the
architecture in a real translation model.

### AGMT English ↔ Vietnamese \~20M

Hugging Face:

**https://huggingface.co/Phitran21/agmt-en-vi-20m**

The Hugging Face repository is the **model demo**, not the canonical
AGMT architecture specification.

It contains the runnable ONNX implementation:

``` text
base.onnx
refine.onnx
router.onnx
decode.onnx
translate.py
```

as well as model-specific:

-   installation instructions;
-   runtime documentation;
-   CLI commands;
-   adaptive inference modes;
-   forced-depth inference;
-   recurrence tracing;
-   qualitative translation examples;
-   known model limitations.

Future AGMT implementations may use different model sizes, language
pairs, or configurations while following the same architecture family.

Conceptually:

``` text
                    AGMT
              Architecture
                   / | \
                  /  |  \
                 ▼   ▼   ▼
              Demo  Future Future
              20M   Model  Model
            EN ↔ VI
```

Therefore the current Hugging Face model should be understood as:

> **an experimental implementation/demo of AGMT, not AGMT itself.**

------------------------------------------------------------------------

# Repository Role

This GitHub repository should be treated as:

``` text
AGMT architecture
+
AGMT training design
+
architecture documentation
+
research/design evolution
```

The Hugging Face repository should be treated as:

``` text
model implementation
+
weights
+
runtime
+
model-specific usage
+
demo
```

This separation allows the AGMT architecture to evolve independently
from any single model release.

------------------------------------------------------------------------

# Current Status

AGMT is currently an **experimental research architecture**.

The architecture should be evaluated independently from the performance
of one small implementation.

A model implementation can fail because of:

``` text
training data
model capacity
optimization
tokenization
decoder limitations
training maturity
```

without automatically establishing that adaptive recurrence itself is
ineffective.

Likewise, successful examples from one model do not by themselves prove
the architecture generally superior.

Evaluation should therefore distinguish:

``` text
architecture behavior
routing behavior
translation quality
semantic capability
compute efficiency
model scale
```

------------------------------------------------------------------------

# Author

**Trần Tuấn Phi --- Vietnam**

AGMT architecture and project creator.

### Demo model

**AGMT EN ↔ VI \~20M**

https://huggingface.co/Phitran21/agmt-en-vi-20m

### Contact

Email: `phihhhhhhhhhh@gmail.com`\
*Email is not checked frequently.*

Facebook:\
https://www.facebook.com/share/1HerWkmghN
