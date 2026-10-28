## ☕ Buy me a coffee

Se ti piace il progetto e vuoi supportarlo, puoi offrirmi un caffè tramite [PayPal](https://paypal.me/nicovic17) oppure [Revolut](https://revolut.me/nicoauriemma).

# Meme Pose Tracker

`run3.py` è un prototipo Python che riconosce alcune pose delle mani tramite webcam e sovrappone automaticamente un meme al volto dell’utente. Il risultato viene mostrato in una finestra locale e pubblicato contemporaneamente come **webcam virtuale OBS**, così può essere usato in applicazioni come Microsoft Teams, Zoom, Discord o Google Meet.

L’elaborazione utilizza MediaPipe per rilevare volto e mani, OpenCV per acquisire e comporre i fotogrammi e `pyvirtualcam` per inviare il video alla webcam virtuale.

## Funzionalità

- rilevamento in tempo reale di un volto e fino a due mani;
- riconoscimento di tre pose tramite i landmark MediaPipe;
- tecnica della sostituzione (Kawarimi no Jutsu): col sigillo della Tigre vieni sostituito da un tronco;
- sovrapposizione del meme associato direttamente sul volto;
- riquadri animati intorno alle mani rilevate;
- stabilizzazione temporale per limitare attivazioni accidentali e sfarfallii;
- indicatore della confidenza del rilevamento;
- anteprima locale a `960 × 540` e 30 FPS;
- pubblicazione dello stesso risultato su OBS Virtual Camera;
- elaborazione completamente locale, senza invio dei fotogrammi a server esterni.

## Pose riconosciute

| Posa | Gesto richiesto | Immagine mostrata |
|---|---|---|
| Due mani sul volto | Due mani aperte, verticali e ravvicinate davanti al volto | `memes/image.png` |
| Mani dietro la testa | Due mani separate ai lati della testa | `memes/memeomg.png` |
| Timeout | Due mani aperte e perpendicolari, disposte a forma di T | `memes/stop.png` |

Le pose hanno priorità diverse per ridurre le sovrapposizioni tra gesti simili. Una posa deve essere riconosciuta per almeno cinque fotogrammi consecutivi prima di attivarsi. Il meme viene rimosso dopo otto fotogrammi senza una posa valida.

## Tecnica della sostituzione

Fai il **sigillo della Tigre**: mani unite davanti al petto, indice e medio di entrambe le mani dritti verso l'alto, anulare e mignolo piegati. Dopo una nuvola di fumo (`memes/nuvole.jpeg`) al tuo posto compare un tronco (`memes/tronco.jpeg`). Rifai il sigillo per ricomparire.

La persona viene separata dallo sfondo con il modello `selfie_segmenter.tflite`. Lo sfondo dietro di te viene imparato mentre ti muovi: subito dopo l'avvio spostati un po' a destra e a sinistra, così la scomparsa risulta più pulita. Le immagini senza trasparenza vengono scontornate automaticamente rimuovendo lo sfondo chiaro.

## Sfondo virtuale

Gli sfondi virtuali di Meet e Zoom riconoscono solo le persone: coprirebbero anche il tronco e il fumo. Per questo lo sfondo virtuale va fatto dallo script e quello dell'app di videochiamata va **disattivato**:

```bash
.venv/bin/python run3.py --sfondo blur                 # sfondo sfocato
.venv/bin/python run3.py --sfondo memes/sfondo.jpg     # immagine a scelta
.venv/bin/python run3.py --sfondo none                 # sfondo reale
```

Senza opzione viene usato `VIRTUAL_BACKGROUND`, definito all'inizio di `run3.py` (predefinito: `memes/background.png`).

## Requisiti

- Windows 10 o Windows 11;
- Python 3 compatibile con MediaPipe;
- una webcam funzionante;
- OBS Studio con il componente **OBS Virtual Camera** installato;
- i modelli MediaPipe `face_landmarker.task` e `hand_landmarker.task`;
- le immagini presenti nella cartella `memes/`.

## Installazione

Apri PowerShell nella cartella del progetto e installa le dipendenze:

```powershell
python -m pip install opencv-python mediapipe numpy pyvirtualcam
```

Se `python` non viene riconosciuto, installa Python e abilita l’opzione per aggiungerlo al `PATH` durante l’installazione.

Installa inoltre OBS Studio. La webcam virtuale deve essere disponibile nel sistema, ma non deve essere già occupata da OBS o da un’altra istanza del programma quando avvii `run3.py`.

## Installazione su macOS

MediaPipe non supporta ancora le versioni più recenti di Python: usa Python 3.12 in un ambiente virtuale (qui con [uv](https://docs.astral.sh/uv/)):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python run3.py
```

Poi:

1. al primo avvio concedi l'accesso alla fotocamera all'app del terminale (**Impostazioni di Sistema > Privacy e sicurezza > Fotocamera**) e riavvia il terminale;
2. per la webcam virtuale installa OBS Studio 30 o successivo (`brew install --cask obs`), aprilo e premi una volta **Avvia videocamera virtuale** per installare l'estensione di sistema, che va approvata in **Impostazioni di Sistema > Generali > Elementi login ed estensioni**. Quindi ferma la videocamera virtuale e chiudi OBS prima di avviare `run3.py`.

Se OBS Virtual Camera non è disponibile, il programma mostra comunque l'anteprima locale.

## Struttura dei file

La cartella deve contenere almeno questi file:

```text
webcam_memes/
├── run3.py
├── face_landmarker.task
├── hand_landmarker.task
├── selfie_segmenter.tflite
└── memes/
    ├── image.png
    ├── memeomg.png
    ├── stop.png
    ├── tronco.jpeg
    ├── nuvole.jpeg
    └── background.png
```

I percorsi vengono calcolati rispetto alla posizione di `run3.py`, quindi il programma può essere avviato anche da una directory di lavoro diversa.

## Avvio

Dalla cartella del progetto esegui:

```powershell
python run3.py
```

Il programma:

1. carica i due modelli MediaPipe e tutte le immagini;
2. prova ad aprire la webcam con DirectShow, Media Foundation e infine con il backend automatico;
3. avvia OBS Virtual Camera a `960 × 540` e 30 FPS;
4. mostra la finestra **Meme Pose Tracker**;
5. analizza volto e mani finché non viene chiuso.

Premi `Q` oppure `Esc` per terminare il programma.

## Utilizzo nelle videochiamate

Dopo aver avviato `run3.py`, apri l’applicazione di videochiamata e seleziona **OBS Virtual Camera** come sorgente video.

Avvia prima `run3.py` e poi l’applicazione che deve ricevere il video. Non avviare contemporaneamente la Virtual Camera dall’interfaccia di OBS, perché un solo programma alla volta può controllare il dispositivo virtuale.

## Personalizzare le immagini

Puoi sostituire un meme mantenendo invariati nome, estensione e posizione del file. Per usare nomi differenti, modifica le costanti all’inizio di `run3.py`:

```python
MEME_PATH = BASE_DIR / "memes" / "image.png"
MEMEOMG_PATH = BASE_DIR / "memes" / "memeomg.png"
STOP_MEME_PATH = BASE_DIR / "memes" / "stop.png"
```

Sono consigliate immagini nitide, con proporzioni adatte al volto. Il programma le ridimensiona mantenendo le proporzioni e le ritaglia quando necessario per non uscire dal fotogramma.

## Configurazione video

Le principali impostazioni sono definite all’inizio di `run3.py`:

```python
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
VIRTUAL_CAMERA_FPS = 30
```

- `CAPTURE_WIDTH` e `CAPTURE_HEIGHT` indicano la risoluzione richiesta alla webcam fisica;
- `CAMERA_WIDTH` e `CAMERA_HEIGHT` definiscono anteprima e webcam virtuale;
- `VIRTUAL_CAMERA_FPS` definisce la frequenza della webcam virtuale.

Ridurre risoluzione o FPS può migliorare le prestazioni sui computer meno potenti. Se modifichi la risoluzione della webcam virtuale, assicurati che sia supportata dalle applicazioni nelle quali vuoi utilizzarla.

## Risoluzione dei problemi

### La webcam non si apre

- chiudi Teams, Zoom, browser e altri programmi che potrebbero utilizzare la webcam fisica;
- controlla in **Impostazioni di Windows > Privacy e sicurezza > Fotocamera** che le applicazioni desktop abbiano il permesso di accedervi;
- prova a scollegare e ricollegare la webcam;
- verifica che la webcam funzioni nell’app Fotocamera di Windows.

### OBS Virtual Camera non si avvia

- verifica che OBS Studio e il componente Virtual Camera siano installati;
- arresta la Virtual Camera dall’interfaccia di OBS, se è già attiva;
- chiudi eventuali altre istanze di `run3.py`;
- riavvia OBS o Windows se il dispositivo virtuale rimane occupato.

### Un modello o un’immagine non viene trovato

Controlla che `face_landmarker.task`, `hand_landmarker.task` e tutti i file nella cartella `memes/` abbiano esattamente i nomi indicati. Il programma interrompe l’avvio se una risorsa obbligatoria non può essere letta.

### Le pose vengono riconosciute male

- usa un ambiente ben illuminato;
- mantieni volto e mani interamente nell’inquadratura;
- evita che le mani si confondano con lo sfondo;
- posiziona la webcam frontalmente;
- esegui il gesto in modo netto e mantienilo per qualche istante;
- evita movimenti troppo rapidi o mani sovrapposte, salvo per i gesti che lo richiedono.

### Il video è lento o scatta

- chiudi le applicazioni che utilizzano molta CPU o GPU;
- riduci `CAPTURE_WIDTH`, `CAPTURE_HEIGHT`, `CAMERA_WIDTH` e `CAMERA_HEIGHT`;
- prova una webcam o una porta USB differente;
- aggiorna i driver della webcam.

## Privacy

I fotogrammi della webcam vengono elaborati localmente. `run3.py` non salva registrazioni e non invia immagini o landmark a servizi online. Il video elaborato viene reso disponibile localmente attraverso OBS Virtual Camera alle applicazioni scelte dall’utente.

Le applicazioni di videochiamata possono applicare proprie condizioni e politiche sulla privacy al flusso ricevuto dalla webcam virtuale.

## Limitazioni

- Il riconoscimento si basa su regole geometriche e può produrre falsi positivi o falsi negativi.
- Le soglie possono richiedere modifiche in base a webcam, distanza, illuminazione e caratteristiche dell’utente.
- Il programma rileva un solo volto e al massimo due mani.
- L’assenza temporanea del volto viene tollerata per circa 1,2 secondi.
- Il progetto è un prototipo sperimentale e non è destinato a impieghi medici, biometrici, di sorveglianza o di sicurezza.

## Supporto al progetto

Se il progetto ti è utile e vuoi sostenerne lo sviluppo, puoi offrire un caffè tramite [PayPal](https://paypal.me/nicovic17) oppure [Revolut](https://revolut.me/nicoauriemma).
