# 🤖 RAG Chatbot - Retrieval-Augmented Generation

Inteligentny chatbot oparty na architekturze **RAG (Retrieval-Augmented Generation)** z automatyczną detekcją stylu odpowiedzi. Projekt wykorzystuje modele językowe do generowania odpowiedzi na podstawie specjalistycznej dokumentacji technicznej.

## 📋 Spis treści

- [Opis projektu](#opis-projektu)
- [Funkcjonalności](#funkcjonalności)
- [Technologie](#technologie)
- [Instalacja](#instalacja)
- [Użycie](#użycie)
- [Architektura](#architektura)
- [Przykłady](#przykłady)
- [Dokumentacja badawcza](#dokumentacja-badawcza)

## 🎯 Opis projektu

RAG Chatbot to inteligentny asystent zaprojektowany do odpowiadania na pytania dotyczące **procedury kalibracji systemu wizyjnego kamera-robot**. System łączy w sobie:

- **Retrieval** - wyszukiwanie semantyczne w bazie wiedzy (FAISS)
- **Generation** - generowanie odpowiedzi za pomocą LLM (Qwen 2.5)
- **Auto-detection** - automatyczne dostosowanie stylu odpowiedzi

### Główne zastosowanie
- Wsparcie operatorów komórek zrobotyzowanych
- Asystent do dokumentacji technicznej
- System Q&A dla procedur przemysłowych

## ⚡ Funkcjonalności

### 🎨 Style odpowiedzi (automatyczna detekcja)

- **Strict** - formalne, dokładne odpowiedzi dla pytań technicznych
- **Casual** - przyjazne, konwersacyjne wyjaśnienia
- **Funny** - humorystyczne odpowiedzi dla pytań poza tematem
- **Vulgar** - wulgarny styl (eksperymentalny)

### 🛡️ Guardrails

- Automatyczna odmowa odpowiedzi dla pytań spoza dokumentu
- Próg podobieństwa (MIN_SIMILARITY = 0.35)
- Deduplikacja powtarzających się fragmentów kontekstu
- Walidacja jakości retrieval (max score tracking)

### 🔍 Zaawansowany Retrieval

- Semantyczne wyszukiwanie z FAISS IndexFlatIP
- Embeddingi wielojęzyczne (multilingual-e5-base)
- Chunking na poziomie zdań
- Top-k retrieval z deduplikacją

## 🛠️ Technologie

| Komponent | Technologia |
|-----------|-------------|
| **Model LLM** | Qwen/Qwen2.5-1.5B-Instruct |
| **Embeddings** | intfloat/multilingual-e5-base |
| **Vector DB** | FAISS (IndexFlatIP) |
| **Framework** | Transformers, Sentence-Transformers |
| **Quantization** | BitsAndBytes (opcjonalnie) |

## 📦 Instalacja

### Wymagania wstępne
- Python 3.8+
- pip / conda

### Kroki instalacji

```bash
# Klonowanie repozytorium
git clone https://github.com/KieltRadek/RAG-Retrieval-Augmented-Generation-.git
cd RAG-Retrieval-Augmented-Generation-

# Instalacja zależności
pip install -U bitsandbytes sentence-transformers faiss-cpu transformers accelerate tf-keras sentencepiece torch numpy
```

**Uwaga**: Na GPU z CUDA można użyć `faiss-gpu` dla lepszej wydajności.

## 🚀 Użycie

### Podstawowe użycie

```python
# Uruchomienie chatbota
python RAG_chatbot.py

# Zadawanie pytań
ask_bot("Ile etapów ma procedura kalibracji?")
ask_bot("Co to jest TCP?")
ask_bot("Kto może wykonywać procedurę kalibracji?")
```

### Parametry funkcji `ask_bot`

```python
ask_bot(question, style="auto")
```

**Parametry:**
- `question` (str): Pytanie do chatbota
- `style` (str): Styl odpowiedzi
  - `"auto"` - automatyczna detekcja (domyślnie)
  - `"strict"` - formalne odpowiedzi
  - `"casual"` - przyjazne wyjaśnienia
  - `"funny"` - humorystyczne odpowiedzi
  - `"vulgar"` - wulgarny styl

### Przykłady pytań

#### ✅ Pytania ON-TOPIC (z dokumentu)
```python
ask_bot("Ile ujęć wzorca należy wykonać dla kamery?")
# → "Zadanie polega na wykonaniu 15 ujęć wzorca..."

ask_bot("Kto odpowiada za szkolenia operatorów?")
# → "Główny Inżynier Wizji odpowiada za szkolenia operatorów."

ask_bot("Jaki jest maksymalny błąd reprojekcji RMS?")
# → "Błąd reprojekcji RMS musi być mniejszy niż 0.3 piksela."
```

#### ❌ Pytania OFF-TOPIC (poza dokumentem)
```python
ask_bot("Ile wynosi prędkość światła?")
# → "Hej, tego nie mam w dokumentach!"

ask_bot("Jak ugotować jajko na twardo?")
# → "Hej! Jako asystent nie jestem w stanie pomóc..."
```

## 🏗️ Architektura

```
┌─────────────┐
│   Pytanie   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  Embedding (E5-base)    │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  FAISS Retrieval (k=3)  │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Score Validation       │
│  (threshold: 0.35)      │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Style Auto-Detection   │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Prompt Construction    │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  LLM Generation         │
│  (Qwen 2.5-1.5B)        │
└──────┬──────────────────┘
       │
       ▼
┌─────────────┐
│  Odpowiedź  │
└─────────────┘
```

### Szczegóły komponentów

#### 1. **Chunking**
```python
def chunk_text(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = [s.strip() for s in sentences if s.strip()]
    return chunks
```
- Dzieli dokument na **pojedyncze zdania**
- Eliminuje puste fragmenty
- Zapobiega powtórzeniom w kontekście

#### 2. **Retrieval**
```python
def retrieve_context(query, k=3):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(q_emb), min(k, len(chunks)))
    # Deduplikacja + zwrócenie max_score
```
- FAISS IndexFlatIP (cosine similarity)
- Top-k retrieval (k=3)
- Zwraca kontekst + max similarity score

#### 3. **Auto-Detection**
```python
def detect_style_automatically(question, context):
    similarity = np.dot(q_emb[0], c_emb[0])
    
    if similarity < 0.3:  # OFF-TOPIC
        return np.random.choice(["funny", "strict"])
    elif has_exclamation:
        return "funny"
    elif is_technical:
        return "casual"
    else:
        return "strict"
```

#### 4. **Generation**
- Model: **Qwen/Qwen2.5-1.5B-Instruct**
- max_new_tokens: **240**
- Temperature:
  - Strict: **0.01** (deterministyczny)
  - Funny/Vulgar: **0.3** (kreatywny)
  - Casual: **0.15** (balansowany)

## 📊 Przykłady działania

### Przykład 1: Pytanie techniczne
```
PYTANIE:
Co to jest TCP?

KONTEKST:
Zatwierdź punkty aby kontroler obliczył dokładną pozycję TCP.
Zamocuj na robocie precyzyjny wskaźnik (igłę) o znanym TCP.

ODPOWIEDŹ (strict):
TCP to pole w układzie, które określa położenie roboty w przestrzeni.
```

### Przykład 2: Pytanie off-topic
```
PYTANIE:
Ile wynosi prędkość światła?

KONTEKST:
[Brak wystarczającego dopasowania]

ODPOWIEDŹ (strict):
Nie znalazłem odpowiedzi w dokumencie.
```

### Przykład 3: Pytanie o szczegóły
```
PYTANIE:
Ile ujęć wzorca należy wykonać dla kamery?

KONTEKST:
Wykonaj serię 15 ujęć wzorca poruszając nim w polu widzenia kamery.

ODPOWIEDŹ (casual):
Zadanie polega na wykonaniu 15 ujęć wzorca. Każdy z nich powinien być
wykonany w różnych kątach widzenia kamery.
```

## 📝 Dokumentacja badawcza

Projekt zawiera szczegółową notatkę badawczą (`notatka_badawcza.txt`) opisującą:

- **Wykonane modyfikacje** - migracja z Bielik 11B na Qwen 2.5
- **Metody testowania** - pytania on-topic, off-topic, absurdalne
- **Fragmenty dialogów** - rzeczywiste odpowiedzi systemu
- **Obserwowane różnice** - wpływ zmian na jakość odpowiedzi
- **Rekomendacje** - optymalizacja dla CPU/GPU
- **Pytania kontrolne** - przygotowanie do obrony projektu

## 🔍 Pytania kontrolne (FAQ)

<details>
<summary><b>Jaki model LLM został użyty i dlaczego?</b></summary>

**Qwen/Qwen2.5-1.5B-Instruct** - mały model (1.5B parametrów), działający efektywnie na CPU/GPU z niskimi wymaganiami, zapewniający sensowne odpowiedzi instruktażowe.
</details>

<details>
<summary><b>Czym jest architektura RAG?</b></summary>

**Retrieval-Augmented Generation** - najpierw wyszukiwanie fragmentów (embedding + FAISS), potem generacja z użyciem tych fragmentów jako kontekstu, co ogranicza halucynacje.
</details>

<details>
<summary><b>Jak działa funkcja retrieve_context?</b></summary>

Embedduje pytanie, szuka w FAISS IndexFlatIP top-k (min(k, liczba chunków)), zwraca unikalne zdania. Parametr **k** określa liczbę fragmentów, **score** mierzy podobieństwo (0-1).
</details>

<details>
<summary><b>Jakie są ograniczenia obecnego podejścia?</b></summary>

- Model może halucynować przy słabym kontekście
- Wysokie MIN_SIMILARITY (0.35) daje więcej odmów
- Brak jeszcze testów jednostkowych
- CPU: wolniejsza generacja przy dużych pakietach testowych
</details>

## 🎓 Rekomendacje

### Dla użytkowników CPU:
- Uruchamiaj pojedyncze pytania
- Zakomentuj pakiet testowy na końcu `RAG_chatbot.py`
- Rozważ model o mniejszej liczbie parametrów

### Dla użytkowników GPU:
- Można zwiększyć `k` w retrieval (np. k=5)
- Użyć `faiss-gpu` zamiast `faiss-cpu`
- Rozważyć większy model (np. Qwen 7B)

### Parametry do tuningu:
- `k=3` w `retrieve_context`
- `MIN_SIMILARITY=0.35` w `ask_bot`
- `max_new_tokens=240`
- `temperature` per styl (strict: 0.01, casual: 0.15, funny: 0.3)

## 📄 Licencja

Projekt stworzony w celach edukacyjnych/badawczych.

## 👤 Autor

**Radek Kielt** - [KieltRadek](https://github.com/KieltRadek)

## 🙏 Podziękowania

- [SpeakLeash](https://huggingface.co/speakleash) - polski model Bielik
- [Qwen Team](https://huggingface.co/Qwen) - model Qwen 2.5
- [FAISS](https://github.com/facebookresearch/faiss) - efektywne wyszukiwanie wektorowe
- [Hugging Face](https://huggingface.co/) - infrastruktura ML

---

⭐ **Jeśli projekt Ci się podoba, zostaw gwiazdkę!** ⭐