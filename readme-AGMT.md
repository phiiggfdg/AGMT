# AGMT v3.2 — Elastic Recurrent Translation Architecture

> **Mục đích của file:** chỉ ghi nhớ **kiến trúc model** và **kiến trúc training**.  
> Không chứa cấu hình kích thước model, số tham số, số layer cụ thể, learning rate, batch size, số epoch, threshold số học hay benchmark result.

---

# 1. Mục tiêu cốt lõi

AGMT là một translator end-to-end có khả năng tự quyết định mức xử lý ngữ nghĩa cần thiết cho từng input.

Nguyên tắc:

> **Translation là mục tiêu cuối. Understanding và compute chỉ là tài nguyên phục vụ translation.**

Model không cố:

```text
hiểu càng sâu càng tốt
```

và cũng không cố:

```text
dùng compute càng ít càng tốt
```

Model phải học:

```text
input dễ
→ xử lý vừa đủ
→ dừng sớm

input trung bình
→ refine thêm khi có ích

input khó
→ được phép refine sâu hơn

input rác / nhãn hỏng
→ không được học rằng "loss cao = cần suy nghĩ nhiều hơn"
```

---

# 2. Kiến trúc tổng thể

```text
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
Shared Recurrent Refiner
  │
  ├─ Translation-Value Router
  │     ├─ HALT
  │     └─ REFINE
  │
  └─ lặp refinement khi còn translation value
  │
  ▼
H_refined
  │
  ▼
Stable Memory Bridge
  │
  ├─ refined stream
  ├─ base contextual stream
  └─ lexical stream
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

---

# 3. Shared Tokenizer

Một tokenizer dùng chung cho hai ngôn ngữ.

Yêu cầu kiến trúc:

```text
shared vocabulary
+
byte-safe fallback
```

Mục tiêu:

- giữ được ký tự hiếm;
- không mất tên riêng;
- không mất mã kỹ thuật;
- không biến Unicode lạ hợp lệ thành unknown;
- tạo không gian embedding chung cho hai chiều dịch.

Tokenizer không được tự động coi:

```text
URL
email
version
IP
model name
technical identifier
emoji
ký hiệu
```

là noise.

---

# 4. Shared Embedding

Source và target dùng chung embedding khi tokenizer cho phép.

Output projection có thể tied với embedding.

Embedding tạo ra:

```text
E
```

Từ `E` tách ra hai hướng:

```text
E
├─ main contextual path
└─ lexical preservation path
```

---

# 5. Lexical Highway

Lexical Highway giữ một representation gần source surface hơn:

```text
E_lex = LexicalProjection(E)
```

Mục đích:

- giữ tên riêng;
- số;
- thuật ngữ;
- code-like token;
- cụm cực ngắn;
- lexical identity;
- những thông tin không cần contextual refinement sâu.

Lexical Highway không phải translator riêng.

Nó chỉ là shortcut representation để information lexical không phải sống sót qua toàn bộ recurrent refinement.

---

# 6. Prelude Encoder

Prelude là phần encoder không recurrent.

Luồng:

```text
E
↓
Prelude
↓
H_base
```

Vai trò:

- tạo contextual foundation;
- xử lý local/global context cơ bản;
- tạo state ổn định cho recurrent refinement;
- tạo minimum usable representation;
- cung cấp `H_base` cho Memory Bridge.

Không hard-code:

```text
Prelude = lexical
Refiner = semantic
```

Specialization phải được học end-to-end.

---

# 7. Shared Recurrent Refiner

Sau `H_base`, model dùng một recurrent refinement module với shared weights.

```text
H_r
  │
  ▼
Shared Refiner
  │
  ▼
H_(r+1)
```

Cùng một transformation được tái sử dụng qua nhiều refinement step.

Mục tiêu:

```text
stored parameters
≠
effective computation depth
```

Một input khó có thể nhận thêm computation mà không cần thêm một bộ parameters riêng cho mỗi depth.

---

# 8. Refinement Step Signal

Shared Refiner phải biết nó đang ở refinement stage nào.

Do đó recurrent state nhận thêm:

```text
step representation
```

Concept:

```text
H_r + StepEncoding(r)
↓
Shared Refiner
↓
H_(r+1)
```

Step signal không chứa target/reference.

Nó chỉ giúp cùng shared block biết refinement state hiện tại.

---

# 9. Recurrent Stability

Refiner phải dùng residual-safe architecture.

Nguyên tắc:

```text
Pre-Norm
+
residual path
+
controlled state update
```

Không để repeated transformation:

- explode hidden norm;
- collapse hidden state;
- oscillate vô ích;
- phá lexical/context information.

Nếu recurrence không tạo depth-quality curve hữu ích, router không được dùng để che lỗi của recurrent core.

---

# 10. Translation-Value Router

Router không trả lời:

```text
"Câu này dài không?"
"Câu này có vẻ khó không?"
"Hidden state thay đổi nhiều không?"
```

Router trả lời:

> **Nếu tiếp tục refinement, translation còn khả năng cải thiện đáng kể không?**

Output conceptual:

```text
remaining_translation_value
```

Decision:

```text
value đủ lớn
→ REFINE

value không còn đáng kể
→ HALT
```

---

# 11. Router phải Source-Only khi Inference

Training có thể dùng target/reference để tạo supervision.

Inference thì không.

```text
TRAIN
source + reference
→ tạo oracle translation-value label

INFERENCE
source representation
→ router prediction
```

Không có target leakage.

---

# 12. Token-Sensitive Router Representation

Router không chỉ dùng mean pooling cả câu.

Một token khó có thể bị chìm trong nhiều token dễ.

Ví dụ:

```text
He sat on the bank of the river.
```

`bank` có thể là token quyết định nghĩa dù toàn câu khá đơn giản.

Router dùng nhiều summary:

```text
Global Summary
+
Hard-Token Summary
+
State-Change Summary
+
Direction / Step / Weak Length Signal
```

---

# 13. Global Summary

Global Summary đại diện trạng thái toàn câu.

Concept:

```text
projected recurrent state
↓
masked pooling
↓
global summary
```

Nó giúp router biết sentence-level context hiện tại.

---

# 14. Hard-Token Summary

Router có một lightweight token scorer.

```text
token states
↓
difficulty/value scoring
↓
weighted hard-token pooling
```

Mục tiêu:

- phát hiện một vài token chưa được giải quyết;
- tránh mean pooling làm mất outlier quan trọng;
- giữ routing ở sequence level nhưng có token sensitivity.

---

# 15. State-Change Summary

Router có thể quan sát mức representation còn thay đổi qua recurrence.

Không đo trực tiếp raw hidden state giữa các layer như một semantic truth.

Trước tiên map state vào một router space dùng chung:

```text
H_r
↓
Router Projection
↓
U_r
```

sau đó mới đo change.

Các summary có thể gồm:

```text
mean change
high-quantile change
top-token change aggregate
variance
```

Change chỉ là feature.

Nó không phải router target.

---

# 16. Length chỉ là Weak Feature

Không có rule:

```text
short → shallow
long → deep
```

Length chỉ là một tín hiệu nhỏ.

Một câu ngắn có thể cần nhiều refinement.

Một câu dài có thể rất literal và dừng sớm.

---

# 17. H_refined

Khi router HALT hoặc chạm compute budget:

```text
H_refined = current recurrent state
```

Đây là contextual anchor chính cho translation.

Không dùng weighted mixture của tất cả recurrence state làm đường chính.

---

# 18. Stable Memory Bridge

Decoder không nhận trực tiếp một recurrent endpoint trần.

Memory Bridge chuẩn hóa ba nguồn:

```text
H_refined
H_base
E_lex
```

Concept:

```text
H_bridge
=
H_refined
+
g_base · P_base(H_base)
+
g_lex · P_lex(E_lex)
```

Sau đó:

```text
Hmem = CodaAdapter(H_bridge)
```

---

# 19. Refined Stream

`H_refined` luôn là anchor.

Nó chứa contextual representation tốt nhất tại refinement depth được chọn.

Memory Bridge không được xóa đường anchor này.

---

# 20. Base Context Shortcut

`H_base` cung cấp contextual information trước recurrent refinement.

Mục đích:

- giữ early context;
- tránh recurrent block phải bảo tồn mọi thông tin qua mọi step;
- giúp câu đơn giản có một đường ổn định;
- giảm distribution shift giữa các recurrence depth.

---

# 21. Lexical Shortcut

`E_lex` cung cấp lexical evidence trực tiếp.

Gate quyết định token nào cần dùng shortcut mạnh hơn.

Không hard-code:

```text
tên riêng → gate = 1
```

Model tự học.

---

# 22. Coda Adapter

Sau Memory Bridge có một adapter nhẹ.

Vai trò:

- chuẩn hóa memory interface;
- giảm distribution shift giữa các recurrence depth;
- đưa representation về format ổn định cho decoder.

Coda không phải thêm một semantic model mới.

---

# 23. Decoder

Decoder là autoregressive Transformer decoder tương đối nông.

Vai trò:

```text
generation
target-language syntax
target history
cross-attention tới Hmem
```

Semantic auxiliary objective không được điều khiển decoder mạnh mặc định.

Encoder/refiner chịu phần lớn contextual representation work.

---

# 24. Bidirectional Translation

Cùng một model dịch:

```text
EN → VI
VI → EN
```

Direction token chỉ target language.

Shared architecture:

```text
Tokenizer
Embedding
Prelude
Refiner
Router
Memory Bridge
Decoder
```

Router có thể nhận translation direction như feature vì hai chiều có difficulty distribution khác nhau.

---

# 25. Inference Modes

Cùng một model có thể có các policy compute khác nhau.

Concept:

```text
Fast
Balanced
Quality
```

Chúng khác nhau ở compute price / halt policy.

Không cần train model riêng cho từng mode.

---

# 26. Force-Deep Fallback

Inference luôn hỗ trợ đường:

```text
force maximum trained refinement
```

Mục đích:

- debug;
- quality fallback;
- compare adaptive vs fixed-depth;
- failure recovery.

---

# 27. Kiến trúc Training Tổng Thể

Training được chia thành các research gate logic, không phải một đống auxiliary loss bật cùng lúc.

```text
DATA SANITATION
      ↓
TRANSLATOR TRAINING
      ↓
RECURRENCE ORACLE
      ↓
FREEZE TRANSLATOR
      ↓
ROUTER LABEL GENERATION
      ↓
ROUTER TRAINING
      ↓
ADAPTIVE CALIBRATION
      ↓
OPTIONAL AUXILIARY EXPERIMENTS
```

---

# 28. Static Data Sanitation

Trước training, chỉ dùng các filter bảo thủ không phụ thuộc model.

Loại:

```text
broken record
missing side
empty side
exact duplicate
clear language mismatch
obvious broken markup
obvious metadata dump
extreme pair mismatch
corrupted text
```

Giữ:

```text
names
numbers
technical text
URLs
versions
mixed-language terminology
symbols
rare Unicode
```

Không cố làm data “sạch tuyệt đối”.

---

# 29. Translation Training trước Router

Router không tham gia optimization ban đầu.

Train translator gồm:

```text
Embedding
Prelude
Shared Refiner
Memory Bridge
Coda
Decoder
```

Primary loss:

```text
token-level translation cross entropy
```

Không semantic loss mặc định.

Không router compute loss.

Không copy loss mặc định.

Không external teacher.

---

# 30. Multi-Depth Translator Training

Translator phải usable ở nhiều recurrent depth.

Có hai strategy hợp lệ để benchmark:

```text
A. Deep-first → elastic
B. Variable-depth from early training
```

Kiến trúc không mặc định một strategy luôn tốt hơn.

Chọn strategy dựa trên:

```text
stability
depth-quality curve
translation quality
```

---

# 31. Deep-First Strategy

Concept:

```text
first:
train a sufficiently refined translation path

then:
introduce shallower recurrence endpoints
```

Mục tiêu:

- cho recurrent transformation học refinement trước;
- tránh shallow objective thống trị quá sớm.

Đây là training strategy candidate, không phải architecture law.

---

# 32. Variable-Depth Strategy

Concept:

```text
sample recurrent depth during training
```

Mục tiêu:

- mọi depth được train từ sớm;
- giảm endpoint distribution mismatch;
- tạo elastic translator trực tiếp.

Đây cũng là candidate.

Deep-first và variable-depth phải được so bằng cùng evaluation framework.

---

# 33. Deep Anchor Path

Trong elastic training nên duy trì một deep/reference path đủ ổn định để:

- giữ translation quality;
- tránh recurrent refiner chỉ tối ưu shallow exits;
- tạo basis cho oracle profiling.

Nhưng:

> **Deepest recurrence không mặc định là best recurrence.**

Teacher/reference depth cuối cùng phải được xác định từ validation/oracle behavior.

---

# 34. Freeze Translator trước Router

Sau khi translator ổn định:

```text
freeze:
Embedding
Prelude
Refiner
Bridge
Coda
Decoder
```

Sau đó mới train router.

Lý do:

```text
translator thay đổi
→ oracle labels thay đổi
→ router target di chuyển
```

Freeze biến router training thành một supervised control problem ổn định.

---

# 35. Recurrence Oracle

Với translator frozen, profile translation tại mọi candidate recurrence.

Cho một sample:

```text
L_0
L_1
L_2
...
L_R
```

`L_r` là target-token-normalized translation loss tại recurrence `r`.

Không dùng raw sentence loss.

---

# 36. Best Achievable Future

Tại recurrence hiện tại `r`, định nghĩa:

```text
future candidates = all valid k > r
```

Không dùng một global anchor duy nhất để tạo target.

Mục tiêu là tìm:

```text
best future translation state
```

cho chính sample đó.

---

# 37. Noise-Aware Best Future

Không dùng raw:

```text
min(L_future)
```

một cách mù quáng.

Lý do:

```text
min của nhiều noisy estimates
→ có downward selection bias
```

Do đó best-future target phải có tolerance.

Concept:

```text
best_future_gain
=
improvement chỉ khi vượt noise margin
```

Những khác biệt quá nhỏ được coi là không đáng thêm compute.

---

# 38. Epsilon-Oracle

Oracle không chọn depth có NLL nhỏ nhất tuyệt đối.

Oracle chọn:

> **recurrent depth nông nhất đã đủ tốt so với best achievable translation của sample.**

Concept:

```text
r_oracle
=
smallest r

such that

L_r <= L_best + ε
```

`ε` đại diện acceptable quality tolerance.

Nó phải được calibration theo translation behavior trên validation, không phải chọn tùy ý.

---

# 39. Router Target

Router học:

```text
remaining useful translation value
```

không học:

```text
absolute loss
```

Concept:

```text
V*_r
=
useful improvement available in future refinement
after removing insignificant/noisy gain
```

Nếu future compute không mang improvement vượt tolerance:

```text
V*_r = 0
```

---

# 40. Robust Router Supervision

Router target phải robust với outlier.

Có thể dùng:

```text
robust regression
gain clipping
trusted-supervision mask
```

Nhưng đây là protection cho controller, không phải một sample-quality model lớn.

---

# 41. Catastrophic-Loss Mask

Một sample có model loss cực bất thường có thể:

- rất khó;
- bị misaligned;
- bị corrupt;
- nằm ngoài training support.

Do đó catastrophic mask chỉ nói:

```text
"không dùng sample này để dạy router"
```

không nói:

```text
"sample này phải bị xóa khỏi translation training"
```

Mask phải bảo thủ.

---

# 42. Router Feature Detach

Router đọc translator state nhưng mặc định không sửa translator.

```text
translator state
↓
stop-gradient
↓
router
```

Lý do:

- controller không làm representation biến dạng để dễ route;
- translation state vẫn tối ưu cho translation.

---

# 43. Router Training Objective

Router học từ frozen-oracle supervision.

Concept:

```text
source-derived features
→ predicted remaining value
```

Loss router nên robust với noisy targets.

Router không nhận reference target làm input.

---

# 44. Compute Price

Sau khi router học translation value:

```text
predicted value
vs
compute price
```

quyết định HALT / REFINE.

```text
value > price
→ refine

value <= price
→ halt
```

Compute price được calibration sau.

Không dùng compute penalty để ép router trước khi router hiểu translation value.

---

# 45. Conservative Halting

False halt nguy hiểm hơn false continue.

Vì vậy:

```text
uncertain / near boundary
→ REFINE
```

mặc định ưu tiên translation quality.

---

# 46. Semantic Capability ≠ Routing Capability

Hai câu hỏi phải được tách:

```text
A. Router có chọn compute hợp lý không?
B. Compute thêm có thật sự giúp model hiểu và dịch đúng nghĩa không?
```

Một router hoàn hảo không chứng minh model hiểu semantic tốt.

---

# 47. Semantic Challenge Evaluation

Architecture phải được đánh giá bằng targeted semantic phenomena, không chỉ corpus BLEU/chrF.

Các nhóm cần test:

```text
lexical ambiguity
idioms
negation
pronouns/coreference
clause relations
word sense
named entities
terminology
numbers/units
short ambiguous phrases
long multi-clause inputs
```

---

# 48. Minimal-Pair Evaluation

Ưu tiên challenge pair chỉ thay một yếu tố semantic.

Ví dụ conceptual:

```text
river bank
vs
financial bank
```

```text
did not say X
vs
said not-X
```

```text
pronoun referring to entity A
vs
entity B
```

Mục tiêu:

```text
refinement sâu hơn
→ semantic accuracy thật sự tốt hơn?
```

không chỉ:

```text
loss thấp hơn?
```

---

# 49. Generated Output Evaluation

Challenge set không nên chỉ dùng contrastive scoring nếu mục tiêu cuối là translation generation.

Phải kiểm tra:

```text
model có thật sự generate đúng output không?
```

Contrastive tests vẫn hữu ích cho diagnosis, nhưng generation correctness quan trọng hơn.

---

# 50. Semantic Auxiliary Learning

Không bật mặc định trong core training.

Chỉ thêm sau khi:

```text
recurrent translator có depth-quality value
AND
router hoạt động
AND
challenge set chỉ ra semantic failure cụ thể
```

---

# 51. Optional Semantic Consistency

Nếu cần, thêm bilingual semantic consistency ở encoder/refiner scope.

Nó phải:

- weight nhỏ;
- không áp mạnh vào decoder;
- qua translation-priority gradient arbitration;
- được ablate riêng.

---

# 52. Optional Selective Distillation

Deep recurrence chỉ làm teacher khi nó thực sự tốt hơn shallow recurrence vượt tolerance.

Không:

```text
deep = teacher chỉ vì deep hơn
```

Mà:

```text
deep improves translation
→ teacher eligible
```

Teacher dùng stop-gradient.

---

# 53. Translation-Priority Gradient Arbitration

Khi auxiliary objective được bật:

```text
translation gradient = authority
```

Nếu auxiliary gradient xung đột translation:

```text
project / suppress conflicting component
```

Auxiliary magnitude cũng phải được giới hạn.

---

# 54. Không dùng Direct Hidden Matching

Không ép:

```text
H_shallow = H_deep
```

Điều này phá specialization.

Nếu distill:

```text
relation
output distribution
semantic compatibility
```

được ưu tiên hơn exact hidden identity.

---

# 55. Noise và Semantic Auxiliary

Sample không đáng tin không được làm semantic teacher mạnh.

Nếu sample bị catastrophic-router mask hoặc static data guard đánh dấu nghi ngờ:

```text
semantic/distillation supervision
→ giảm hoặc bỏ
```

Translation training có thể vẫn giữ nếu sample chưa bị static filter loại.

---

# 56. Long-Context Architecture

Long-context robustness gồm ba vấn đề riêng:

```text
data coverage
position representation
available recurrence
```

Không được nhầm:

```text
câu dài lỗi
→ chắc chắn cần thêm recurrence
```

Nếu forced-deep vẫn lỗi:

```text
kiểm tra data / position / tokenizer / decoder
```

---

# 57. Position Architecture

Position mechanism phải được giữ orthogonal với recurrent-depth claim.

Baseline và AGMT phải dùng cùng position mechanism khi so core architecture.

Sau đó mới ablate:

```text
absolute/sinusoidal
relative position
rotary-style position
```

---

# 58. No Silent Truncation

Training/evaluation không được cắt source/target rồi vẫn coi sample là hoàn chỉnh.

Long input phải có policy rõ:

```text
supported
excluded
segmented
```

Không silent truncate.

---

# 59. Short-Input Architecture

Câu cực ngắn có ba đường bảo vệ:

```text
Lexical Highway
H_base shortcut
early HALT
```

Mục tiêu:

```text
không bắt "Hello." phải chạy toàn bộ recurrent depth
```

nhưng nếu input ngắn mà ambiguity cao, router vẫn được phép refine.

---

# 60. Technical / Named-Entity Robustness

Tên riêng và technical spans không được xem là noise mặc định.

Lexical Highway + byte-safe tokenizer là first-line mechanism.

Copy objective chỉ là optional extension nếu challenge set chứng minh cần.

---

# 61. Optional Data Quality Models

Không nằm trong core architecture/training.

Chỉ thêm nếu experiment chứng minh noise vẫn là bottleneck.

Ví dụ concept:

```text
parallel adequacy scoring
model-based bitext filtering
cross-lingual sentence similarity
```

Không gộp chúng vào AGMT core claim.

---

# 62. Optional Curriculum

Không nằm trong core.

Nếu cần test:

```text
uniform shuffled training
vs
curriculum
```

Curriculum không được dùng để che recurrent-training instability.

---

# 63. Optional Subword Regularization

Không nằm trong core.

Có thể test để tăng robustness của:

- rare words;
- names;
- technical terms;
- segmentation variation.

Inference vẫn deterministic.

---

# 64. Training Architecture — Compact Form

```text
┌──────────────────────────────┐
│ STATIC DATA SANITATION       │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ TRAIN TRANSLATOR             │
│                              │
│ Translation CE only          │
│ Multi-depth recurrence       │
│ No router control            │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ FREEZE TRANSLATOR            │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ PROFILE ALL VALID DEPTHS     │
│                              │
│ Build ε-oracle               │
│ Build best-future targets    │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ TRAIN ROUTER                 │
│                              │
│ Source-only features         │
│ Robust value regression      │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ CALIBRATE HALTING POLICY     │
└───────────────┬──────────────┘
                ▼
┌──────────────────────────────┐
│ EVALUATE                     │
│                              │
│ Compute behavior             │
│ Translation quality          │
│ Semantic challenge sets      │
└───────────────┬──────────────┘
                ▼
        OPTIONAL AUXILIARIES
```

---

# 65. Model Architecture — Compact Form

```text
Source
  │
  ▼
Shared Byte-Safe Tokenizer
  │
  ▼
Shared Embedding ──────► Lexical Highway
  │                         │
  ▼                         │
Prelude                     │
  │                         │
  ▼                         │
H_base ─────────────────────┤
  │                         │
  ▼                         │
Shared Recurrent Refiner    │
  ▲       │                 │
  │       ▼                 │
  └── Router:               │
      HALT / REFINE          │
          │                  │
          ▼                  │
      H_refined ─────────────┘
          │
          ▼
      Memory Bridge
          │
          ▼
      Coda Adapter
          │
          ▼
   Shallow AR Decoder
          │
          ▼
      Translation
```

---

# 66. Quy tắc Architecture

1. **Translation là final authority.**
2. **Understanding không phải mục tiêu độc lập.**
3. **Compute được cấp theo expected translation value.**
4. **Không dùng sentence length làm luật depth.**
5. **Short input có lexical/base shortcuts.**
6. **Hard input được phép dùng thêm recurrence.**
7. **Stored parameters và effective depth được tách bằng recurrent sharing.**
8. **Router source-only khi inference.**
9. **Router không được train trước khi translator có depth-quality curve.**
10. **Translator được freeze trước khi tạo router supervision cuối.**
11. **Oracle chọn shallowest depth đủ tốt, không chọn deepest/best-NLL một cách mù quáng.**
12. **Best-future gain phải có tolerance chống noise.**
13. **High absolute loss không đồng nghĩa hard input.**
14. **Data lạ nhưng hợp lệ không được coi là rác.**
15. **Memory Bridge luôn giữ refined anchor.**
16. **Không all-layer over-mixing mặc định.**
17. **Semantic auxiliaries chỉ bật sau khi core được chứng minh.**
18. **Deep teacher chỉ hợp lệ khi deep computation thực sự cải thiện translation.**
19. **Không silent truncation.**
20. **Compute quality và semantic quality phải được đánh giá riêng.**

---

# 67. Quy tắc Training

1. **Static filtering trước model-based filtering.**
2. **Core translator train bằng translation loss trước.**
3. **Multi-depth behavior phải được học trong translator.**
4. **Deep-first và variable-depth là training alternatives cần benchmark.**
5. **Không cho router làm translator target di chuyển.**
6. **Freeze translator trước router training cuối.**
7. **Profile recurrence curve trên frozen translator.**
8. **Dùng ε-oracle để định nghĩa "đủ tốt".**
9. **Router học remaining useful translation value.**
10. **Router supervision phải robust với noisy outlier.**
11. **Near-boundary inference ưu tiên refine.**
12. **Compute penalty chỉ xuất hiện sau khi router có meaningful value estimate.**
13. **Semantic/distillation không được che core failure.**
14. **Auxiliary gradient không được phá translation gradient.**
15. **Challenge-set semantic evaluation là bắt buộc để kiểm tra "hiểu".**

---

# 68. Ý tưởng trung tâm cuối cùng

AGMT không cố tạo:

```text
small LLM
+
translator
```

AGMT cố tạo:

> **một translator duy nhất có khả năng dùng lượng latent computation phù hợp với lợi ích dịch thực tế của từng input.**

Với input đơn giản:

```text
lexical/base representation đã đủ
→ dừng
```

Với input cần context:

```text
refinement còn translation value
→ tiếp tục
```

Với input rất khó:

```text
được phép dùng nhiều recurrent compute hơn
```

Với input rác:

```text
không được mặc định hiểu loss cao là "cần suy nghĩ sâu"
```

Với mọi input:

```text
translation quality
>
semantic elegance
>
compute saving
```

Theo nghĩa:

- translation là mục tiêu cuối;
- semantic representation chỉ có giá trị khi giúp translation;
- compute saving chỉ có giá trị khi không phá translation.

---

# 69. Một câu tóm tắt

> **AGMT v3.2 = shared recurrent translator + translation-value routing + lexical/base shortcuts + stable memory bridge + frozen-oracle router training, trong đó model chỉ refine khi compute thêm có giá trị dịch thực sự và dừng ở depth nông nhất đã đủ tốt.**
