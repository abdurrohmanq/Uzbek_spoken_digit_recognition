# O'zbekcha raqamlarni ovozdan tanish (log-mel spektrogramma + CNN + Gradio demo)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abdurrohmanq/uzbek-spoken-digit-recognition/blob/main/uzbek_spoken_digit_recognition.ipynb)

O'zbek tilida aytilgan 0–9 raqamlarini audiodan aniqlaydigan model. Har bir audio log-mel spektrogrammaga aylantiriladi va kichik CNN bilan 10 sinfli klassifikatsiya qilinadi. Natijani sinab ko'rish uchun Gradio'da mikrofon yoki `.wav` fayl bilan ishlaydigan web sahifa qilindi.

## Loyiha haqida qisqacha

| | |
|---|---|
| **Vazifa** | Ko'p sinfli audio klassifikatsiya (10 ta raqam) |
| **Belgilar** | Log-mel spektrogramma: 16 kHz, 1 soniya, 64 mel, 0–4 kHz → `(64, 101, 1)` |
| **Model** | 3 blokli CNN (Conv2D + BatchNorm + MaxPooling), Keras |
| **Dataset** | 6 spikerning 1 024 ta yozuvi → augmentation bilan 4 096 ta |
| **Kutubxonalar** | `librosa`, `soundfile`, `tensorflow` / `keras`, `scikit-learn`, `matplotlib`, `gradio` |
| **Muhit** | Google Colab, GPU (Tesla T4) |
| **Demo** | Gradio web interfeys (`share=True` bilan ochiq havola) |

## Dataset

- `audio_uz.zip` — 1 024 ta `.wav` fayl, nom formati: `{raqam}_{spiker}_{namuna}.wav`
- 6 ta spiker, har bir raqam bo'yicha deyarli teng miqdorda
- Yozuvlar mono, uzunligi 0.5–3.3 soniya (median 1.3 s)
- **Yozuvlar har xil sample rate'da**: 8 kHz dan 48 kHz gacha. Bu birinchi urinishdagi asosiy muammo bo'lib chiqdi (pastga qarang).

Audio fayllar real odamlarning ovozi bo'lgani uchun repoga qo'shilmagan. Notebook arxivni `gdown` bilan yuklab oladi.

### Augmentation

Har bir original yozuvdan 3 ta qo'shimcha nusxa yaratildi:

- oq shovqin qo'shilgan
- ikki xil pitch shift (`librosa.effects.pitch_shift`)

Jami: 1 024 × 4 = **4 096** ta yozuv.

### Train / test bo'linishi

| To'plam | Qaysi yozuvlar | Soni |
|---|---|---|
| Test | Har bir spikerning 0 va 1-namunalari (augmented nusxalari bilan) | 440 |
| Train | Qolgan barcha yozuvlar | 3 656 |
| Validation | Train'ning 20% i (`validation_split=0.2`) | — |

## Ish jarayoni

1. **Yuklash va augmentation.** Arxiv `gdown` bilan yuklanadi, ochiladi va har bir yozuvga shovqin hamda pitch shift qo'llanadi.
2. **Diagnostika.** Original va augmented fayllarning sample rate'i, kanal soni va davomiyligi tekshiriladi.
3. **Log-mel spektrogramma.** Har bir audio bir xil holatga keltiriladi:

   ```python
   y, _ = librosa.load(path, sr=16000)              # 16 kHz, mono
   y, _ = librosa.effects.trim(y, top_db=25)        # jimlikni kesish
   # 1 soniyaga keltirish: qisqa bo'lsa nol qo'shiladi, uzun bo'lsa kesiladi
   mel = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=512,
                                        hop_length=160, n_mels=64, fmax=4000)
   logmel = librosa.power_to_db(mel, ref=np.max)    # (64, 101)
   ```

   `fmax=4000` tanlangan, chunki 8 kHz yozuvlarda 4 kHz dan yuqori chastota umuman yo'q. Shu bilan barcha yozuvlar bir xil chastota oralig'ida solishtiriladi.

4. **Normallashtirish.** Mean va std faqat train'dan olinadi va test'ga ham shu qiymatlar qo'llanadi (data leakage bo'lmasligi uchun).
5. **CNN model:**

   ```
   Input (64, 101, 1)
   Conv2D(32)  → BatchNorm → MaxPool
   Conv2D(64)  → BatchNorm → MaxPool
   Conv2D(128) → BatchNorm → MaxPool
   Flatten → Dense(128, relu) → Dropout(0.3) → Dense(10, softmax)
   ```

6. **O'qitish.** Adam, `sparse_categorical_crossentropy`, batch 32, ko'pi bilan 30 epoch, `EarlyStopping(patience=5, restore_best_weights=True)`.
7. **Baholash.** `classification_report`, confusion matrix va har bir spiker bo'yicha alohida aniqlik.
8. **Gradio demo.**

## Birinchi urinish: nega model hamma audioga "3" dedi

Birinchi versiyada spektrogrammalar `matplotlib` bilan 64×64 PNG rasm qilib saqlanib, CNN rasmlar ustida o'qitildi. Natija:

- 30 epoch davomida train accuracy **≈ 0.10**, loss **≈ 2.302**
- 2.302 = ln(10) — bu 10 sinfning har biriga 10% ehtimol beradigan, ya'ni **hech narsa o'rganmagan** modelning loss'i
- Model har bir audioga bir xil ehtimollar berdi va `argmax` har safar "3" ni tanladi

Sabablar:

1. Spektrogramma funksiyasi 8 kHz audio uchun yozilgan edi (`Fs=2`, chiziqli chastota o'qi). Rasm har doim 0 dan Nyquist chastotasigacha cho'zilardi. Natijada bir xil raqam 8 kHz va 48 kHz yozuvda butunlay boshqacha ko'rinardi.
2. Augmented fayllar `librosa.load` default'i bo'yicha 22 050 Hz da yozilgan, originallar esa boshqa sample rate'da qolgan.
3. PNG orqali o'tishda ma'lumot yo'qolgan: 8-bit, har bir rasmga alohida kontrast, 3 ta bir xil RGB kanal.

Yechim: audio 16 kHz ga keltirildi va spektrogramma PNG emas, sonli matritsa sifatida modelga berildi.

| | Eski (PNG) | Yangi (log-mel) |
|---|---|---|
| Test accuracy | 0.10 | **0.79** |
| Bashoratlar | hammasi "3" | 10 ta raqamga taqsimlangan |

## Web demo

```python
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath",
                    label="Raqamni ayting yoki .wav yuklang"),
    outputs=gr.Label(num_top_classes=3, label="Bashorat"),
    examples=examples,          # har bir raqamdan bittadan, har xil spikerlardan
    title="O'zbekcha raqamlarni tanish (0–9)",
)
demo.launch(share=True)
```

`predict_digit` audioni train'dagi bilan aynan bir xil tayyorlaydi: log-mel → train'ning mean/std si bilan normallashtirish → model. `share=True` vaqtinchalik ochiq `*.gradio.live` havola beradi, u faqat Colab runtime ishlab turganda ishlaydi.

## Natijalar

| Ko'rsatkich | Qiymat |
|---|---|
| Validation accuracy | ≈ 0.95 |
| Test accuracy (110 ta original yozuv) | **0.79** |

Validation natijasi optimistik: train'dagi yozuvlarning shovqinli va pitch-shift nusxalari validation'ga ham tushadi. Test natijasi esa halol, chunki u boshqa namunalardan iborat.

### Spikerlar bo'yicha aniqlik (test, original yozuvlar)

| Spiker | Fayllar | Aniqlik |
|---|---|---|
| A (8 kHz yozuv) | 20 | 0.40 |
| B | 20 | 0.90 |
| C | 20 | 0.75 |
| D | 20 | 0.85 |
| E | 20 | 0.95 |
| F | 10 | 1.00 |
| **Jami** | **110** | **0.79** |

Umumiy aniqlik yaxshi ko'rinsa ham, 8 kHz da yozilgan spikerda model sezilarli darajada yomon ishlaydi.

## Duch kelingan muammolar va yechimlar

| Muammo | Sabab | Yechim |
|---|---|---|
| Model hamma audioga "3" dedi, accuracy 0.10 | PNG spektrogrammalar 8 kHz uchun yozilgan funksiyada yasalgan, sample rate'lar aralash | 16 kHz log-mel, PNG'siz |
| 8 kHz yozuvlarda spektrogrammaning yuqori yarmi bo'sh | 8 kHz audioda 4 kHz dan yuqori chastota yo'q | `fmax=4000` |
| Validation 0.95, test esa 0.79 | Validation'ga train yozuvlarining augmented nusxalari tushadi | Test natijasi yakuniy baho sifatida olindi |
| Gradio misollari hammasi bitta spikerdan edi | Fayllar tartiblanganda birinchi spikerniki tushgan | Misollar uchun spikerlar navbat bilan almashtirildi |
| Mikrofon orqali yozilgan ovozda model adashdi | Yozuvlar 5–8 s, bir nechta so'zdan iborat va past ovozda. Model esa faqat birinchi 1 soniyaga qaraydi | Bitta raqamni qisqa aytish kerak; to'liq yechim — keyingi qadamlarda |

## Cheklovlar

- Atigi 6 ta spiker. Test ham shu 6 kishining boshqa yozuvlari, shuning uchun 0.79 **yangi ovozdagi** aniqlikni ko'rsatmaydi.
- Mikrofon orqali yangi ovozda sinalganda model ko'pincha adashadi.
- Model faqat bitta raqamni va faqat 1 soniyani ko'radi.
- Gradio havolasi vaqtinchalik: Colab sessiyasi yopilsa, demo ham to'xtaydi.

## Keyingi qadamlar

- Gradio'da eng baland 1 soniyalik bo'lakni olish va model ko'rayotgan spektrogrammani natija yonida ko'rsatish
- Spiker bo'yicha baholash: har safar bitta spikerni butunlay testga qoldirish (leave-one-speaker-out)
- Ko'proq spikerdan yozuv yig'ish (20–30+ kishi)
- SpecAugment, tezlikni o'zgartirish va 8 kHz ga tushirib-ko'tarish augmentation'lari
- Oldindan o'qitilgan audio model (wav2vec2 yoki Whisper encoder) ustiga kichik klassifikator o'qitish
- Modelni saqlab, demoni Hugging Face Spaces'da doimiy joylashtirish

## Ishga tushirish

1. Notebookni Google Colab'da oching (yuqoridagi tugma).
2. **Runtime → Change runtime type → T4 GPU** ni tanlang.
3. Data yuklash va augmentation kataklarini ishga tushiring.
4. PNG spektrogrammali birinchi urinish bo'limini o'tkazib yuborish mumkin. Yangi pipeline **"Yangi pipeline: log-mel spektrogramma"** bo'limidan boshlanadi.
5. Kataklarni ketma-ket ishga tushiring. Gradio katagi ochiq havolani chiqaradi.

```bash
pip install librosa soundfile tensorflow scikit-learn matplotlib gradio gdown
```

## Repo tuzilishi

```
uzbek-spoken-digit-recognition/
├── uzbek_spoken_digit_recognition.ipynb
└── README.md
```
