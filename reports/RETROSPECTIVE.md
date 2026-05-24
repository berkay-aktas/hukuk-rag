# RETROSPECTIVE — Turkish Legal RAG (`hukuk-rag` / CENG493)

> Bu rapor `~/.claude/history.jsonl`'den kurtarılan 512 user prompt'una (17 session, 18 Mart → 23 Mayıs 2026), git commit geçmişine ve `~/.claude/projects/.../memory/`'deki otomatik memory dosyalarına dayanılarak yazıldı. Claude'un cevapları korunmadığı için bazı kararların "neden"i sadece Berkay'ın pasted text'leri ve memory'deki sonuç notlarından rekonstrükte edilebildi.
> Yeni PC'de bu repo clone edildiğinde Claude session transkriptleri **gitmiş** olacak — bu doküman onun yerine geçer.

---

## GÜVENLİK — ÖNCELİKLE BUNU OKU

History dosyasında **canlı bir Kaggle API key** plaintext olarak gözüküyor:

> `[03-19 01:07]` — **"KGAT_75851e321e1251c9aa1722ce4cf8542e first this is my api key for kaggle..."**

Aksiyon:
1. Kaggle hesabına gir → Account → API → **Expire API Token**, yenisini oluştur
2. `reports/c493_history_recovered.md` dosyasını yeni PC'ye kopyalıyorsan o satırı **redact** et (key'i `KGAT_<redacted>` yap). Bu dosya repoda olmasın — `.gitignore`'a ekle ya da repo dışında tut
3. Yeni token'ı tekrar plaintext yazma; Colab Secrets üzerinden enjekte et (zaten commit `224884f` ile bu standardı uygulamıştın — yeni token için de aynısı geçerli)

---

## 1. Genel timeline ve mega-session özeti

Toplam 17 session var ama gerçek iş **`21fff18a`** numaralı tek bir mega-session içinde geçti — 18 Mart 22:17'de açıldı, 13 Nisan'a kadar canlı kaldı, sonra 23 Mayıs'ta soğuk start. Diğer 16 session sadece `/resume`, `/exit`, `/mcp` kısa session'ları — yani Berkay aslında tüm projeyi **tek bir konuşma ipliğinden** yürüttü, PC her crash'lediğinde aynı session'a geri döndü.

**Kabaca fazlar:**

| Tarih aralığı | Faz | Ne yapıldı |
|---|---|---|
| **18 Mart 22:17 → 19 Mart 04:30** | Açılış + proje tanımı | Hocanın audio transcription'ı, dataset doğrulama, CLAUDE.md ilk versiyonu, repo adı tartışması, gitignore, ilk Kaggle/HF download denemeleri (Colab session 3 kere crash'ledi) |
| **19 Mart 19:52 → 20 Mart 05:54** | Colab MCP onboarding + baseline encoding | `colab-mcp` server kurulumu, T4 üzerinde 1.06M chunk için baseline E5 encoding (2h 35dk, sonra FAISS IVF-PQ + BM25 inşa edildi) |
| **20 Mart 15:30 → 21 Mart 03:00** | Fine-tuned embedding training + repo cleanup | L4 GPU'ya geçiş, triplet'lerle e5 fine-tuning, "PROJECT_PLAN.md commit → 1 saat sonra revert" (no md files kuralı), README yazımı |
| **21 Mart 02:50 → 22 Mart 04:14** | FT FAISS rebuild (1h+ crash, yeniden başlatma) + gold set inşası | 2.49M chunk encoding, mevzuat.gov.tr cross-check ile gold set 10→140→225'e büyütüldü, 7406/7499/7550 yasa değişiklikleri elle entegre edildi |
| **22 Mart 13:00 → 23 Mart 04:30** | C4 QLoRA + C1/C2/C4/C5 ablation | QLoRA 5h 32dk eğitim, C4'ün ilk çıktısı tamamen bozuk ("KastenDemocratsence, kasten复工复%@\",死去"), retraining, sonra 225 soruluk eval'ları sırayla |
| **23 Mart 16:00 → 24 Mart 08:43** | C5/C6 ablation tamamlama + standardized prompt re-runs | Tüm config'ler aynı prompt template'iyle yeniden çalıştırıldı (confound düzeltmesi) |
| **(boşluk: 24 Mart → 12 Nisan)** | Sessizlik | İki haftadan fazla hiç prompt yok — muhtemelen ara verildi |
| **12 Nisan 19:50 → 13 Nisan 03:58** | Progress report PDF + retrieval-only metrics + statistical analysis | Sanity check (5 soru pipeline trace), retrieval metric bug fix (`base_ids` alignment), Wilcoxon + bootstrap CI'lar, progress report v2/v3 iterasyonları |
| **23 Mayıs 09:27 → 09:44** | Geri dönüş, "geçmişi göster" | "bu folderda hic rag ile alakali chat yok mu, gecmise bakmam lazim" — yani bu retrospektifi tetikleyen an |

Toplam aktif iş günü ~6 (Mar 18-23) + 1 (Apr 12-13). Geri kalan zaman crash recovery, MCP reconnect ve compute unit bekleme.

---

## 2. Ne tatmin etti

### 2.1 Embedding fine-tuning kararı (C2)
Tek başına `+19% Token F1, p=0.0025` veren bileşen oldu. Memory'de bu açıkça **"BEST"** olarak işaretli ve "Embedding FT is highest-impact single component" notu var. Yani üzerinde en çok zaman harcadığı şey (1100k → 2.49M chunk rebuild, L4 üzerinde 5+ saat) gerçekten geri döndü.

### 2.2 Standardized prompt across configs (confound düzeltmesi)
23 Mart'ta C1–C5'i aynı prompt template'iyle yeniden çalıştırma kararı:
> `[03-23 15:47]` — *"save the old C1–C3 numbers alongside the new ones. Don't overwrite. In your report, you can include a small table showing 'before vs after prompt standardization' — it demonstrates methodological rigor"*

Berkay bunu beğendi ve uyguladı. Bu bir hoca-impressing methodological win.

### 2.3 Colab MCP entegrasyonu
İlk gün manuel "copy paste token" döngüsü yüzünden delirme noktasına geldi:
> `[03-19 02:18]` — *"isnt there a colab mcp? im tired of re copy pasting tokens every time"*

`colab-mcp` server'ı kurulduktan sonra workflow çığlık çığlığa hızlandı. Tüm ileri günler MCP üzerinden yürüdü.
**WHY:** Colab serbest tier'da session her crash'lediğinde token yeniden geliyor; manuel akışta saatte 5-6 kere takıldı.
**NEXT TIME:** Yeni PC'de ilk gün `colab-mcp` kurulumu ve Colab Secrets onboarding'i. Manuel token döngüsüne ASLA dönme.

### 2.4 Gold set'i 10 → 140 → 225'e büyütme
Başlangıçta 10 soruluk template vardı. Berkay law student arkadaşlarını Google Sheets ile organize etti, sonra mevzuat.gov.tr cross-check ile her satırı yasal değişikliklere göre güncelletti (7406, 7499, 7550 sayılı kanunlar). 225-question gold set commit `f1e9c93` ile repo'ya girdi.
> `[03-21 04:54]` — *"for eg it says ... but in actual there are some additions"* — bu, LLM'in yarattığı paraphrase'lerin mevzuat metnini düşürdüğünü gördüğü andı

**Bu cidden işe yaradı:** Tüm 225-question eval'ları aynı sabit set üzerinde koştu, yani config'ler arası fair karşılaştırma var. Memory'de "**NEVER train on this**" notu da bu disiplinin bir parçası.

### 2.5 Granular conventional commits, "no co-author, no claude mention"
> `[03-19 00:30]` — *"add claude.md to commit granularly and conventional. dont add claude as co author. dont mention claude in pr. dont commit claude.md to repo."*

`git log --oneline`'a bakınca her commit gerçekten 1 mantıksal değişiklik (fix: skipna, fix: Counter overlap in token_f1, refactor: shared turkish_tokenize, vb.). Bu disiplin profesyonel görünüm üretti — hoca için "hangi gün ne yaptın" sorusuna anlık cevap veriyor.

### 2.6 Statistical analysis notebook (`03_statistical_analysis.ipynb`)
Wilcoxon signed-rank + bootstrap 95% CI'lar tüm 6 config'in tüm 3 metrik'inde mevcut. Memory'deki **"Embedding effect (C1→C2): *** on all three metrics (p<0.003)"** gibi notlar bunun PhD-level bir analiz olmasını sağlıyor.

---

## 3. Ne tatmin etmedi / neyi yanlış yaptı

### 3.1 Çok erken büyük training başlatması, sonra crash → toplam ~10 saat israfı

İlk full corpus encoding 18-19 Mart gecesi başlatıldı, **3 ayrı crash** yaşandı:
> `[03-19 01:35]` — *"it crashed 3 times"*
> `[03-19 01:40]` — *"it crashed again just do it yourself and give me the link and i run ok?"*
> `[03-19 01:50]` — *"it crashed again amk"*

Sonra 20-21 Mart'ta 1100k satırdan başlatılan FT encoding gece ortasında error verdi:
> `[03-21 13:23]` — *"IT JUST ERRORED AFTER ONE HOUR"*
> `[03-21 13:24]` — *"I JUST WOKE UP"*

**WHY:** Crash resilience'ı sonradan inşa etti (commit `2ef0d30` "rewrite baseline notebook for crash resilience"), ama crash'lerin yarısı çoktan olmuştu. Streaming chunker, skip-if-exists, hardcoded paths (commit `89053d2` *"make all notebook cells self-contained with hardcoded paths"*) — hepsi reaktif düzeltmeler.

**NEXT TIME:** Yeni PC'de **ilk 100 chunk ile end-to-end smoke test** yapmadan büyük encoding başlatma. Mega-batch checkpoint logic'ini Day 1'de yaz.

### 3.2 PC'nin (kendi PC'sinin) Colab session'ı düşürmesi

> `[03-20 17:54]` — *"my pc shut down!!!"*
> `[03-23 00:05]` — *"MY PC SHUT DOWN!!! CAN YOU CONNECT AGAIN PLEASE"*
> `[03-23 18:00]` — *"MY PC CRUSHED MID C2 RUN AND SCRATCHPAD IS ALL GONE, NOW HOW AM I GONNA FIND THE PREV RUN OH NOOO I DIDNT SAVE THE COPY:(("*

Defalarca yaşandı. Drive'a checkpointing iyi düzeyde çalıştı (CLAUDE.md'de **"Checkpoint everything to Google Drive every 500 training steps"** kuralı var) ama Colab notebook'unun kendisi browser tab'ı kapanınca cache'den kayboldu.

**NEXT TIME:**
- Colab notebook'unu **File → Save a copy in Drive** ile periyodik manuel snapshot
- PC sleep ayarlarını "never" yap (sadece monitor off)
- Önemli runlar başlarken `notebook.ipynb`'i de repo'ya `git push` et (output'larla beraber)

### 3.3 İlk PROJECT_PLAN.md commit'i, sonra silindi

Commit history net gösteriyor:
- `26f9a31` — **"docs: add 8-week implementation plan with ablation configs"** (Mar 20, 23:10)
- `3dda0f0` — **"chore: remove implementation plan from repo"** (Mar 21, 00:09)

Yani 1 saat sonra geri çekti. Sebep:
> `[03-21 00:09]` — *"no needed to include plan :D i told you this before no md files and dont mention 8 week"*

**WHY:** Memory'deki `feedback_preferences.md` *"Do NOT create markdown files unless explicitly asked"* + *"Do NOT mention course name, week numbers, or academic context"* kuralları zaten vardı ama Claude unuttu.
**NEXT TIME:** Yeni Claude session'ı başında bu kuralı **explicit hatırlat**: "no md files in repo, no week mentions, no course mentions, no co-author".

### 3.4 QLoRA C4 ilk training tamamen garbage output verdi

5 saat 32 dakikalık QLoRA training'in çıktısı:
> `[03-22 23:37]` PASTED #48 — *"Answer: KastenDemocratsence, kasten复工复%@\",死去; meklüme'',死去..."*

> `[03-22 23:39]` — *"run 3 agents team to decided which option is more effective based on the plan. its okay we have still a lot time and we can retrain i can buy more units even if i dont want to. **i dont want to try and lose again, anything to win this contest please**"*

Diagnoz `[03-23 02:23]`'te geldi: *"This is likely a gradient_checkpointing + use_cache conflict at inference time"*. İkinci training'de 1h sürede düzgün çalıştı.

**WHY:** İlk training'de `model.gradient_checkpointing_enable()` aktifken inference'ta `use_cache=True` çakıştı, çıktı bozuldu.
**NEXT TIME:** QLoRA training sonunda **ilk şey** inference smoke test — 1 prompt, oku, garip karakter var mı? Sonra eval başlat. 2h eval'ın ortasında "demek bozukmuş" demek geri dönüşü zor.

### 3.5 Cross-encoder reranker (C3) sistematik olarak performansı bozdu

Memory: *"Reranker consistently harmful across all metrics (p<0.0001)"* — yani **fine-tuned reranker tüm metric'lerde anlamlı şekilde DAHA KÖTÜ**. C6 (everything combined) en kötü config oldu (Token F1 0.1180, baseline'dan %16 düşük).

Kök neden memory'de yazılı: *"52% of training queries had no positive passages, so the model trained on noisy silver labels"*.

**WHY:** Reranker training set'ini construct ederken pozitif passage'ı garanti edemedi — yarısı pure noise üzerinde fine-tune oldu.
**NEXT TIME:** Bir sonraki iterasyonda **ya off-shelf reranker'ı bırak**, ya da training data'yı manuel pozitif label'la (en az 1k clean örnek). Silver label hiçbir zaman yeterli değildi.

### 3.6 Tekrar tekrar açıklamak zorunda kaldığı şeyler

Aynı tercihi 3+ kez söyledi:

- **"granular commits, conventional, no co-author"** — Mar 19 00:30, Mar 20 23:09, Mar 23 02:00 (en az 3 kez)
- **"don't run cells while encoding is running"** — Mar 20 23:00 (*"dont add cell or something this time while its running"*), Mar 20 16:27 (*"why did you stop the running code you said let it cook"*)
- **"don't suggest go sleep / rest"** — Mar 22 03:05 (*"dont suggest me to go rest or go bed please one more time please:)"*), Mar 23 04:42 BÜYÜK HARFLE: *"NEVER GO FOR A SHORTCUT BECAUSE OF TIME CONSTRAINTS, ALWAYS ASSUME THERES ENOUGH TIME..."*

Bu üç tercih şu an memory'de yazılı (`feedback_preferences.md`) ama yeni PC'de fresh memory ile bunlar **gidecek**. Bölüm 7'deki opening prompt bunu kapsıyor.

### 3.7 Compute unit yönetimi

> `[03-19 22:46]` — *"Cannot connect to GPU backend. You cannot currently connect to a GPU due to usage limits"*
> `[03-21 04:36]` — *"Available: 69.68 compute units. Usage rate: approximately 1.71 per hour WILL IT BE EVEN ENOUGH?"*
> `[03-22 13:06]` — *"we have 32 compute unit left btw"*
> `[03-22 23:37]` — *"we have 20 computing units left XDD you said it will leave 70 :D"*
> `[03-23 03:27]` — *"Available: 14.28 compute units"*
> `[03-23 15:47]` — *"WE HAVE 5 COMPUTING UNIT LEFT SHOULD I BUY 100 MORE?"*

İki kere 100 unit satın aldı. **Tahmin hatası**: Claude "70 unit kalır" dedi, 20 kaldı. Memory'de *"Colab Pro = aynı fiyat, çok daha iyi"* tartışması da var.

**NEXT TIME:** Direkt **Colab Pro+** abonelik, pay-as-you-go değil. Aylık $50 ama background execution + 500 unit + uzun runtime. PAYG hesaplaması tutmuyor çünkü crash yüzünden 2x harcıyorsun.

### 3.8 Knowledge graph dropped (geç fark edildi)

Memory'deki `feedback_final_report.md`'de: *"Knowledge graph — DROP unless time is abundant. Muddies the 'embedding FT dominates' narrative."*

Ama README'de hâlâ *"NetworkX (KG)"* ve *"+ Knowledge graph + citation chains"* yazıyor (C6 satırında). README ile gerçek state mismatch.

**NEXT TIME:** README'yi memory'deki gerçek state'e göre düzelt — KG section'ını çıkar ya da "deferred to future work" yaz.

---

## 4. Bu sefer farklı yapması gereken şeyler (somut, eylem maddesi)

| # | Aksiyon | Sebep |
|---|---|---|
| 1 | **Day 1'de Kaggle token rotate + Colab Secrets** | Eski token leaked (yukarıda) |
| 2 | **Colab Pro+ subscription**, PAYG değil | Crash yüzünden 200+ unit eridi, Pro+ aynı para çok daha iyi |
| 3 | **Notebook'u Drive'a periyodik Save a Copy + repo'ya output ile push** | "MY PC CRUSHED" anlarında scratchpad'in gitmesini engelle |
| 4 | **Büyük encoding/training başlatmadan önce 100-örnek smoke test** | 3'lü crash kayıpları, garbage QLoRA output (5h israfı) |
| 5 | **QLoRA training sonrası ilk iş inference smoke** (1 prompt, garip karakter check) | C4'ün ilk denemesi 2h eval'ın yarısında bozuk fark edildi |
| 6 | **Reranker'ı silver label ile retrain etme** — ya off-shelf bırak ya da min. 1k clean pozitif | C3/C6 sistematik olarak baseline'dan kötü, p<0.0001 |
| 7 | **README'yi gerçek state ile sync et** — KG çıkar, C5-C8 wishful description'ları memory'deki gerçek sonuçlarla değiştir | README "agentic system" diyor ama gerçek state 6 config ablation |
| 8 | **`reports/generate_progress_report.py` ve `reports/c493_history_recovered.md` `.gitignore`'da kalsın** | Memory'de *"NEVER commit reports/generate_progress_report.py. User explicitly requested this."* + recovered history'de exposed key var |
| 9 | **Standardized prompt'u Day 1'den uygula** — tüm config'ler aynı template | İlk runlarda config-spesifik prompt'lar kullandı, sonra confound düzeltmesi için her şeyi 2h × 5 config = 10h yeniden çalıştırdı |
| 10 | **Notebooks 01/02/03 dışında 04 (reranker noise curve) + 05 (Gradio demo) hâlâ TODO** | Memory roadmap'inde required deliverable olarak yazılı |

---

## 5. Saklanması gereken kararlar (Why bilgisi ile)

### 5.1 Embedding fine-tuning > QLoRA single-impact-wise
- **Karar:** E5'i Turkish legal triplets ile contrastive fine-tune et (intfloat/multilingual-e5-large üzerine)
- **WHY:** Memory: *"+19% Token F1 vs +7% for QLoRA"*. Embedding tek başına en büyük lever.
- **NEXT TIME:** Eğer kısıtlı zamanın varsa, tek bileşene fokus = embedding.

### 5.2 IVF-PQ FAISS (Flat değil)
- **Karar:** `IndexIVFPQ` (nlist=256, m=32, nbits=8, nprobe=16) — config.yaml'de yazılı
- **WHY:** 2.49M chunk × 1024-dim float32 = ~10GB Flat index, RAM'e sığmaz. PQ ile %95+ recall, 10x küçük.
- **NEXT TIME:** Train sample boyutuna dikkat — `index.train()` reprezantatif sample gerekli, atlama.

### 5.3 RRF fusion (k=60), weighted ensemble değil
- **Karar:** `rrf_merge(dense, bm25, k=60)` — `src/retrieval/fusion.py`
- **WHY:** Score scale farklarını normalize etmiyor — rank-based, robust. Weighted'da BM25 score'ları her zaman dense'i bastırırdı.

### 5.4 Token F1 + ROUGE-L + BERTScore — üçü birden raporlanır
- **Karar:** Üçünü de tüm config'lerde bootstrap CI + Wilcoxon ile
- **WHY:** Memory: *"Token F1 understates quality for Turkish legal QA due to agglutinative morphology. BERTScore shows the true picture."* + *"QLoRA NOT significant on Token F1 (p=0.45) but *** on BERTScore (p=0.001)"*.
- **NEXT TIME:** Eğer sadece Token F1 raporlarsan QLoRA "işe yaramadı" gözükür. Üçünü beraber ver.

### 5.5 480-token chunks (512 değil)
- **Karar:** `max_tokens: 480, overlap_tokens: 64` (config.yaml)
- **WHY:** CLAUDE.md'de pitfall: *"multilingual-e5-large max sequence length is 512 tokens. The 'passage: ' prefix + tokenizer special tokens eat ~5-10 tokens. If your chunks are 512 tokens, they WILL get silently truncated."*

### 5.6 Legal-aware chunking — `Madde \d+` boundary
- **Karar:** Önce `Madde` numarasına göre böl, sonra paragraph, en son token-based fallback
- **WHY:** Bir maddenin ortasından kesmek anlamı parçalıyor, retrieval citation'ı bozuyor.

### 5.7 Two publication-ready contributions
Memory `feedback_final_report.md`'de:
1. *"Surface-overlap retrieval metrics fail to capture the quality of contrastively fine-tuned embeddings in RAG"*
2. *"Silver-labeled reranker training degrades below a quantifiable noise threshold"*

Bu iki claim'i merkezi tut, her şey buna hizmet etsin. KG, agentic vs. saçma dağılma.

### 5.8 No co-author, conventional commits, no md files in repo
Yukarıda 3. bölümde açıklandı. Memory'de ve CLAUDE.md'de yazılı, ama yeni session her seferinde unutuyor.

---

## 6. Kayıp giden context — yeni PC'de elinde olmayacaklar

Claude responses transkripti hiç korunmamıştı; sadece Berkay'ın prompt'ları ve pasted Claude output'ları var. Aşağıdakiler **sonsuza kadar kayıp**:

1. **C4 QLoRA'nın garbage output debug session'ı** — `gradient_checkpointing` vs `use_cache` çakışmasının tam diagnozu, fix sırası, hangi config flag'iyle düzeltildi (memory'de sadece "*This is likely a gradient_checkpointing + use_cache conflict*" notu var, gerçek fix kodu Drive'daki notebook'ta)

2. **Base FAISS index alignment bug** — *"Base FAISS index had different chunk ordering than chunk_text_list"* memory'de yazılı ama nasıl detect edildi (self-retrieval test) ve fix tam olarak ne idi (`base_ids[idx]["text"]` kullan), bu konuşma kayıp. Eğer base index'i yeniden buildersen aynı bug'a düşersin.

3. **3-agent decision team output'ları** — `[03-20 18:28]` ve `[03-22 23:39]`'da "run 3 agents team to decide" dedi. O agent'lerin verdiği reasoning kayıp.

4. **Mevzuat.gov.tr cross-check workflow detayları** — Hangi maddelere hangi yasa değişikliği uygulandı (7406, 7499, 7550) — gold set JSON'unda son hali var ama "neden bu eklendi" history kayıp.

5. **Compute unit'leri nasıl harcadığı** detaylı timeline (hangi run kaç unit, total ~200 unit muhtemelen)

6. **Drive `/content/drive/MyDrive/hukuk-rag/` içinde ne var** — `models/`, `indexes/`, `results/` altında tam dosya isimleri. Yeni PC'den Drive'a tekrar mount olunca bunu **listele ve döküman**la.

### Manuel toparlama listesi (yeni PC'de Day 1):

- [ ] Google Drive'a mount → `/content/drive/MyDrive/hukuk-rag/` altındaki tüm dosyaları listele, `reports/drive_inventory.txt` olarak kaydet
- [ ] `indexes/` altındaki `bm25.pkl`, `faiss_base.index`, `faiss_ft.index`, `chunk_text_list.pkl`, `base_ids.pkl`, `ft_ids.pkl` — alignment'ı sanity-check et (chunk[0]'ı arat, kendi index'ine düşüyor mu)
- [ ] `models/checkpoint-10000` (FT E5) ve `models/qlora-c4` adapter'ı yüklenip test edilebiliyor mu
- [ ] `results/config{1-6}*.json` dosyalarını oku, memory'deki metric'lerle eşleşiyor mu
- [ ] QLoRA `gradient_checkpointing` fix'ini bir notebook cell'inde **kodla yorum olarak** sabitle (memory'den çıkar, kod içine koy)

---

## 7. Bir sonraki session için açılış prompt önerisi

Yeni PC'de `git clone` sonrası Claude Code başlattığında **ilk mesaj olarak** bunu yapıştır:

````
Selam. Bu repo Turkish Legal RAG (CENG493 term project), `berkay-aktas/hukuk-rag`.
Yeni PC, fresh memory — eski session transkriptleri yok.

ÖNCE BUNLARI OKU (sırayla):
1. CLAUDE.md (kök) — project conventions
2. reports/RETROSPECTIVE.md — eski sessions'ın özeti, ne tatmin etti / etmedi
3. README.md — mimari özet (DİKKAT: KG kısmı outdated, gerçekte drop edildi)
4. configs/config.yaml — tüm hyperparam'lar
5. data/gold/gold_test_set.json — 225-question held-out set (ASLA TRAIN ETME)

SABİT TERCİHLER (her oturum unutuluyor, lütfen disipline uy):
- Conventional commits, GRANULAR (her commit 1 değişiklik), NO co-author line, NO "claude" mention
- NO new .md files (RETROSPECTIVE.md ve README.md yeterli)
- NO course/week mention in repo files
- Notebook'lar Drive checkpoint'lerine dayanır; commit'e output'larıyla beraber push
- Asla "go rest / go sleep" deme; zaman kısıtı yok varsay
- pnpm değil pip, Python 3.10+, T4/L4 Colab, QLoRA 4-bit her zaman

PROJE STATE (memory'den):
- 6 config ablation tamam (C1-C6, 225 soru, standardized prompt)
- BEST: C2 (FT Embed only) — Token F1=0.1678, BERTScore=0.6680, p<0.003
- WORST: C6 (everything) — Token F1=0.1180, reranker bozdu (p<0.0001)
- Bootstrap CI + Wilcoxon tüm pair'lerde mevcut
- Progress report PDF reports/ altında

DRIVE STATE (varsayım — DOĞRULA):
/content/drive/MyDrive/hukuk-rag/
  indexes/ (faiss_base, faiss_ft, bm25.pkl, chunk_text_list.pkl)
  models/checkpoint-10000 (FT E5)
  models/qlora-c4 (QLoRA adapter — gradient_checkpointing fix uygulanmış olmalı)
  results/config{1-6}*.json

ŞU AN YAPACAĞIM:
[BURAYA NE YAPMAK İSTEDİĞİNİ YAZ — örn:
 "reranker noise curve experiments (10/30/50/70%) — memory'deki stretch goal"
 ya da
 "Gradio demo — required deliverable, henüz yok"
 ya da
 "final report — 10-15 sayfa technical paper"]

İlk adım: Drive'ı mount et, indexes/ inventory'sini çıkar, sonra ne yapacağıma karar verelim.
````

Bu prompt 3 şey yapıyor: (1) memory'den kayıp olan tercihleri rebuild ediyor, (2) gerçek state'i özetliyor, (3) Claude'a "Drive'da ne var bilmiyorum, önce kontrol et" diyerek hayali asumptions'a düşmesini engelliyor.

---

## Kapanış notu

Bu projenin en güçlü tarafı **disiplinli ablation + statistical rigor** (Bootstrap CI, Wilcoxon, 3 metrik, 225-question fixed gold set). En zayıf tarafı **reactive crash recovery + reranker silver-label trap**. Yeni iterasyonda fokus: (1) reranker'ı düzgün clean data ile retrain ya da bırak, (2) Gradio demo + final report (required deliverable, hâlâ yok), (3) opsiyonel reranker noise curve (PhD-level stretch contribution).

İki publication-ready bulguya sadık kal: "*retrieval metrics fail for contrastive FT embeddings*" + "*silver-label reranker degradation threshold*". Bunlar dağılırsa rapor zayıflar.
