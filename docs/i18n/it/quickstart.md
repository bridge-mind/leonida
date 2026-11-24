# Leonida — Guida rapida (`it`)

> [!NOTE]
> La documentazione in inglese ([`README.md`](../../../README.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) e
> [`docs/README.md`](../../README.md)) è la **fonte di riferimento**. Questa guida raccoglie solo le istruzioni
> stabili di configurazione e compilazione (sincronizzata con il commit `49475796`). Per lo stato quotidiano del
> progetto consulta il [`README.md`](../../../README.md).

## Che cos'è Leonida

Leonida è un progetto di ricompilazione statica di Grand Theft Auto VI (Xbox Series) per eseguirlo nativamente su
Windows, **costruito sul tuo PC a partire dalla tua copia**. Il codice del gioco viene caricato ed eseguito
nativamente (la CPU di Xbox Series è x86-64 come quella di un PC), il software di sistema della console è
reimplementato in shim e i programmi GPU RDNA2 della console vengono tradotti in SPIR-V per Vulkan. Modelli,
texture e audio sono letti dal tuo pacchetto.

> [!IMPORTANT]
> Questo repository **non contiene il gioco, chiavi né eseguibili**, e non spiega come ottenerli. Nulla qui
> decifra o aggira protezioni. Tutto ciò che la build produce (shader tradotti, asset estratti, cartella di
> lavoro e salvataggi) deriva dalla tua copia e **deve restare solo sul tuo computer**: non allegare né
> condividere questi file in issue o pull request (vedi [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)).

Il progetto è in fase iniziale: **il gioco non si avvia ancora**.

## Prerequisiti

- **La tua copia** del pacchetto Xbox Series, versione `1.0.2.1` (la cartella che contiene `MicrosoftGame.Config`
  e gli archivi `.rpf` come file normali su disco).
- **Windows 10/11 x64** con circa **400 GB liberi** per i file di lavoro (possono stare su un altro disco con
  `--work`).
- [Python 3.12+](https://www.python.org/downloads/) (spunta «Add python.exe to PATH»).
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).
- [Vulkan SDK](https://vulkan.lunarg.com/), per `spirv-as` / `spirv-val` (e `vulkaninfo`).
- [LLVM 21+](https://github.com/llvm/llvm-project/releases) con il **target AMDGPU**, per `llvm-mc`:
  - La build ufficiale di LLVM per Windows **non** include il target AMDGPU.
  - Per compilare un `llvm-mc.exe` nativo con AMDGPU, esegui questo comando dalla radice del repository
    (non serve WSL):

    ```powershell
    powershell -ExecutionPolicy Bypass -File scripts\build_llvm_mc.ps1
    ```

    Lo script richiede `git` e gli strumenti C++ di Visual Studio (vedi sotto), imposta `LLVM_MC` per il tuo
    utente e ti chiede di aprire un nuovo terminale per caricare la variabile.
  - In alternativa, installa `llvm-21` in WSL (Ubuntu) e imposta la variabile con il comando completo, nella
    stessa sessione PowerShell da cui avvierai `make_game.py`:

    ```powershell
    $env:LLVM_MC = 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'
    ```

    Vedi [`docs/design.md`](../../design.md#disassembling-shaders) per i dettagli.
  - Su Linux basta avere `llvm-mc` nel `PATH`.
- **Visual Studio 2022** con il carico di lavoro **Sviluppo di applicazioni desktop con C++** e il **Windows SDK
  `10.0.26100` o successivo**, necessari per la fase `runtime` (il runtime usa le root signature versione 1.2,
  che gli SDK più vecchi non dichiarano).
- [ffmpeg](https://ffmpeg.org/download.html) nel `PATH` *(facoltativo: decodifica gli stream XMA2 ed è
  obbligatorio con `--wav`)*.

## Come compilare

### Modo semplice (Windows)

1. Installa i prerequisiti qui sopra.
2. Scarica il progetto **con il suo sottomodulo**:

   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```

   Il «Download ZIP» di GitHub non include `third_party/rpf9-explorer`. La build lo scarica da sola se `git` è
   installato, altrimenti ti dice cosa fare.
3. Fai doppio clic su **`Build.bat`**.

Lo script cerca da solo la tua copia del gioco: Download, Desktop, Documenti (anche dove Windows li sposta, per
esempio in OneDrive), Giochi, la cartella utente e la radice di ogni disco, fino a due livelli di profondità.
Riconosce la versione da poche dimensioni di file, senza leggere nulla di grande. Se hai più copie ti chiede
quale usare, e se non ne trova nessuna ti chiede di incollare il percorso della cartella. Prima di partire
verifica che tutto il necessario sia installato e che ci sia abbastanza spazio libero, e ti elenca in un colpo
solo ciò che manca. Il risultato va in `Documents\Leonida`. Se la build viene interrotta, riprende da dove si era
fermata.

### Dal terminale (`make_game.py`)

Stesso comportamento, con più controllo:

```powershell
python -I make_game.py --find                          # elenca le copie trovate su questo PC
python -I make_game.py --pick                          # scegli la cartella del gioco in una finestra
python -I make_game.py --check                         # verifica la tua copia, senza compilare
python -I make_game.py --check --full-check            # confronta anche lo SHA-256 di ogni file (lento)
python -I make_game.py --game "D:\percorso\del\pacchetto" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets         # solo alcune fasi
python -I make_game.py --game "D:\percorso\del\pacchetto" --wav   # decodifica anche Opus in PCM WAV (grande)
python -I make_game.py --wave64-emulate off            # auto (predefinito), on oppure off
```

Le fasi vengono eseguite in quest'ordine: `check` → `shaders` → `assets` → `runtime` → `pipelines`.
`pipelines` crea ogni shader tradotto sul driver della tua GPU e salva una cache delle pipeline, così al primo
avvio non devono essere compilati. La traduzione degli shader è il passaggio più lungo: usa `--jobs` processi
(per impostazione predefinita metà dei thread della CPU).

- **`--wave64-emulate`**: traduce gli shader compute e mesh wave64 in modo che siano esatti con qualunque
  dimensione di subgruppo. Le GPU NVIDIA e Intel hanno subgruppi da 32 lane e ne hanno bisogno. Il valore
  predefinito `auto` interroga `vulkaninfo` (parte del Vulkan SDK) e attiva l'emulazione a meno che ogni GPU
  non dichiari subgruppi da 64, oppure quando non riesce a stabilirlo. Resta spenta quando la GPU del runtime è
  una scheda o iGPU RDNA2 (RX 6000, grafica integrata Ryzen 6000/7000/9000) con il driver AMD per Windows, che
  esegue wave64 in modo nativo. La modalità viene salvata accanto agli shader tradotti, e cambiarla li traduce
  di nuovo. Vedi «Wave64 emulation» in [`docs/design.md`](../../design.md).
- **Opzioni di ripiego** (inoltrate al traduttore, salvate accanto agli shader; cambiarle li traduce di nuovo):
  `--no-float-controls`, `--bary-fallback` (GPU senza `VK_KHR_fragment_shader_barycentric`),
  `--texel-offset-fallback` (GPU senza `VK_KHR_maintenance8`) e `--mesh-compute` (GPU senza
  `VK_EXT_mesh_shader`). Sono documentate in [`docs/design.md`](../../design.md).
- **`--work`**: conviene puntarlo a un disco con almeno 400 GB liberi.
- **`--wav`**: disattivato di default perché il WAV PCM a 16 bit occupa molto più dell'Ogg Opus.
- **Linux e Wine/Proton**: `make_game.py` funziona anche su Linux. Per la fase `runtime` fuori da Windows passa
  `--runtime-build <cartella>` con una build Windows (`gta6rt.exe` e `gta6rt_tls_host.dll`, per esempio
  l'artefatto `runtime-windows` della CI scompattato): lo script copia i file in `--out` e scrive `gta6rt.sh`,
  che avvia `gta6rt.exe` con `wine` (o con il Wine indicato dalla variabile `WINE`; per Proton, il suo
  `files/bin/wine`). La stessa opzione salta la compilazione anche su Windows. In alternativa, su Linux la fase
  `runtime` può cross-compilare il loader con clang-cl e lld-link (LLVM 19+, esclusi 22.1.7 e 22.1.8) contro
  l'SDK di Windows preparato da [xwin](https://github.com/Jake-Shadle/xwin): vedi
  [`docs/testing.md`](../../testing.md#the-windows-runtime-on-linux-under-wine-or-proton).

## Problemi comuni

- **Hai scaricato con «Download ZIP» di GitHub**: lo ZIP non include il sottomodulo `third_party/rpf9-explorer`.
  Usa `git clone --recursive` oppure esegui `git submodule update --init --recursive` nella cartella del
  repository.
- **`llvm-mc` rifiuta `gfx1013` / AMDGPU**: probabilmente hai l'installer ufficiale di LLVM per Windows. Segui
  una delle opzioni LLVM descritte nei prerequisiti.
- **Errore su `D3D12_ROOT_SIGNATURE_DESC2` durante la build del runtime**: aggiorna il Windows SDK alla versione
  `10.0.26100` o successiva dal Visual Studio Installer oppure con
  `winget install Microsoft.WindowsSDK.10.0.26100`, elimina la cartella di build e riconfigura.
- **Spazio esaurito sul disco `C:`**: passa `--work "D:\leonida-work"` per spostare i file temporanei su un'altra
  partizione.

## Come contribuire ed eseguire i test GPU

1. Controlla le pull request aperte e l'[issue di coordinamento][coord] per evitare lavoro duplicato. Riserva
   l'attività scrivendo `claiming: <ID>` (oppure una descrizione di una riga se non esiste ancora un ID) prima
   di aprire la pull request.
2. Esegui i test pertinenti con input sintetici. Per esempio, per il verificatore dei manifest:
   `python -I tests/test_manifest_verifier.py` dalla radice del repository. La suite del traduttore si esegue
   con `python -m unittest discover -s tests/semantic -v`.
3. Nella pull request indica l'ID dell'attività, cosa hai provato e, se hai eseguito test GPU, quale GPU e
   quale driver. Le regole sono in [`CONTRIBUTING.md`](../../../CONTRIBUTING.md). Se hai usato un assistente
   IA, dichiaralo nella pull request, come richiesto da [`AGENTS.md`](../../../AGENTS.md).

**Mai** allegare a issue o pull request file del gioco, asset estratti o shader generati: condividi solo
risultati (conteggi, esito dei test, classi di errore).

La [guida ai test GPU](../../testing.md) spiega come eseguirli senza i file del gioco. I risultati dei test GPU
vanno pubblicati nell'[issue #166](https://github.com/ItsNotRexed/Leonida/issues/166).

[coord]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22
