# Leonida — Szybki start (`pl-PL`)

> [!NOTE]
> Dokumentacja w języku angielskim ([`README.md`](../../../README.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) i
> [`docs/README.md`](../../README.md)) jest **wersją obowiązującą**. Ten przewodnik zawiera tylko stabilne instrukcje
> konfiguracji i budowania (zsynchronizowany z commitem `f847748c`). Aktualny stan projektu znajdziesz w
> [`README.md`](../../../README.md).

## Czym jest Leonida

Leonida to projekt statycznej rekompilacji Grand Theft Auto VI (Xbox Series), który uruchamia grę natywnie na
Windowsie. Port jest budowany **na Twoim komputerze z Twojej własnej kopii gry**. Procesor Xbox Series jest x86-64
tak jak w komputerze PC, więc kod gry jest ładowany i wykonywany natywnie, a oprogramowanie systemowe konsoli jest
implementowane od nowa w postaci tzw. shimów. Skompilowane wcześniej programy GPU konsoli (RDNA2) są tłumaczone na
SPIR-V (Vulkan), a modele, tekstury i audio są odczytywane bezpośrednio z Twojej paczki.

> [!IMPORTANT]
> To repozytorium **nie zawiera gry, kluczy ani pliku wykonywalnego** i nie wyjaśnia, jak je zdobyć. Nic tutaj nie
> odszyfrowuje danych ani nie obchodzi zabezpieczeń. Wszystko, co tworzy build (przetłumaczone shadery, wyodrębnione
> zasoby, folder roboczy i zapisy gry), pochodzi z Twojej kopii i **musi zostać wyłącznie na Twoim komputerze** —
> nigdy nie załączaj ani nie udostępniaj tych plików w issues ani pull requestach (zobacz
> [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)).

Projekt jest na wczesnym etapie: **gra jeszcze się nie uruchamia**.

## Wymagania

- **Twoja własna kopia** paczki Xbox Series w wersji `1.0.2.1` (folder z plikiem `MicrosoftGame.Config` i archiwami `.rpf` czytelnymi na dysku jako zwykłe pliki).
- **Windows 10/11 x64** z około **400 GB wolnego miejsca** na pliki robocze (mogą trafić na inny dysk dzięki `--work`).
- [Python 3.12+](https://www.python.org/downloads/) (zaznacz *„Add python.exe to PATH”*).
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).
- [Vulkan SDK](https://vulkan.lunarg.com/) (`spirv-as`, `spirv-val` i `vulkaninfo`).
- [LLVM 21+](https://github.com/llvm/llvm-project/releases) z **targetem AMDGPU** (`llvm-mc`):
  - Oficjalna wersja LLVM dla Windowsa **nie** zawiera targetu AMDGPU.
  - Aby zbudować natywny `llvm-mc.exe` z AMDGPU, uruchom `.\scripts\build_llvm_mc.ps1` w PowerShellu w głównym
    folderze repozytorium (WSL nie jest potrzebny). Skrypt wymaga `git` i narzędzi C++ z Visual Studio, ustawia
    zmienną `LLVM_MC` dla Twojego użytkownika i prosi o otwarcie nowego terminala, żeby zmienna była widoczna.
  - Jeśli używasz WSL, zainstaluj `llvm-21` w Ubuntu i ustaw zmienną w tej samej sesji PowerShella, w której
    uruchamiasz `make_game.py`:

    ```powershell
    $env:LLVM_MC = 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'
    ```

    Więcej szczegółów w [`docs/design.md`](../../design.md#disassembling-shaders).
  - Na Linuksie wystarczy `llvm-mc` w `PATH`.
- **Visual Studio 2022** z pakietem *Programowanie aplikacji klasycznych w języku C++* i **Windows SDK `10.0.26100`
  lub nowszym** (potrzebne do etapu `runtime`, który używa root signatures w wersji 1.2; starsze SDK ich nie
  deklarują).
- [ffmpeg](https://ffmpeg.org/download.html) w `PATH` *(opcjonalnie: dekoduje strumienie XMA2 i jest wymagany przy
  `--wav`)*.

**Mało miejsca na dysku:** około 400 GB dotyczy pełnego builda. Do *uruchomienia* portu etapy
`--stages check,shaders,runtime` potrzebują około 30 GB; etap `assets` (~370 GB) służy tylko narzędziom do zasobów,
a `--wav` dodaje kilkakrotność rozmiaru Ogg Opus. `--work` i `--out` mogą leżeć na innym dysku; paczka na dysku
tylko do odczytu wymaga `--overlay` (opcja `gta6rt`) na dysku zapisywalnym. Nie usuwaj plików `.s` z
`work/shaders/dxbc/disasm`: skasowanie ich z pozostawieniem folderu sprawia, że etap wygląda na ukończony z zerem
shaderów, a `disasm` będący dowiązaniem symbolicznym jest odrzucany.

## Jak zbudować

Portu nie pobiera się gotowego. Buduje się go na Twoim komputerze z Twojej kopii.

**Prosty sposób:**

1. Zainstaluj wymagania powyżej.
2. Pobierz projekt **razem z submodułem**:
   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```
   „Download ZIP” z GitHuba pomija `third_party/rpf9-explorer`. Build pobierze go sam, jeśli masz `git`, a w
   przeciwnym razie powie, co zrobić.
3. Kliknij dwukrotnie **`Build.bat`**.

Skrypt sam znajdzie Twoją kopię gry: szuka w Pobranych, na Pulpicie, w Dokumentach (także tam, gdzie Windows je
trzyma, np. w OneDrive), w Games, w folderze użytkownika i na początku każdego dysku, do dwóch poziomów w głąb.
Rozpoznaje wersję po rozmiarach kilku plików, bez czytania dużych danych. Jeśli znajdzie kilka kopii, zapyta, której
użyć, a jeśli żadnej — poprosi o wklejenie ścieżki do folderu. Przed startem sprawdza, czy wszystko jest zainstalowane
i czy jest dość wolnego miejsca, i wymienia wszystkie braki naraz. Wynik zapisuje w `Documents\Leonida`. Przerwany
build można uruchomić ponownie — będzie kontynuowany od miejsca, w którym się zatrzymał.

### Z terminala (`make_game.py`)

To samo zachowanie i więcej kontroli:

```powershell
python -I make_game.py --find                          # lista kopii gry znalezionych na tym komputerze
python -I make_game.py --pick                          # wybór folderu gry w oknie dialogowym
python -I make_game.py --check                         # sprawdza Twoją kopię, niczego nie buduje
python -I make_game.py --check --full-check            # dodatkowo porównuje SHA-256 każdego pliku (wolno)
python -I make_game.py --game "D:\sciezka\do\paczki" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets         # tylko wybrane etapy
python -I make_game.py --game "D:\sciezka\do\paczki" --wav  # dodatkowo dekoduje Ogg Opus do PCM WAV (duże pliki)
python -I make_game.py --wave64-emulate off            # auto (domyślnie), on lub off
```

Etapy wykonują się w tej kolejności: `check` → `shaders` → `assets` → `runtime` → `pipelines`. Etap `pipelines`
tworzy każdy przetłumaczony shader na sterowniku Twojej karty graficznej i zapisuje cache potoków, żeby pierwsze
uruchomienie nie musiało ich kompilować. Tłumaczenie shaderów trwa najdłużej i domyślnie używa połowy wątków
procesora (można to zmienić przez `--jobs`).

- **`--wave64-emulate`**: decyduje, czy shadery compute i mesh wave64 są tłumaczone tak, by były dokładne przy
  dowolnym rozmiarze podgrupy (zobacz „Wave64 emulation” w [`docs/design.md`](../../design.md)). Karty NVIDIA i
  Intel mają podgrupy 32-wątkowe i potrzebują tej emulacji. Domyślny tryb `auto` pyta `vulkaninfo` (część Vulkan SDK)
  i włącza emulację, jeśli którejś karcie brakuje podgrup 64-wątkowych albo nie da się ustalić ich rozmiaru. Sprawdza
  też, czy karta używana przez runtime potrafi uruchamiać shadery compute i mesh z podgrupami 64-wątkowymi; z
  `--mesh-compute` wystarczy obsługa compute. W sterowniku AMD dla Windowsa `auto` używa natywnego wave64 na RDNA1 i
  RDNA2, gdy karta runtime to obsługuje — nawet jeśli inna karta ma podgrupy 32-wątkowe albo `vulkaninfo` jest
  niedostępne; ten sterownik potrafi się wysypać na emulowanym kodzie LDS (#857). Na kartach RDNA1 bez obsługi mesh
  decyzja opiera się na etapie compute, który potrafi uruchomić wave64 bez natywnego mesh. Tryb jest zapisywany obok
  przetłumaczonych shaderów, a jego zmiana tłumaczy je ponownie. `auto` zwykle może użyć ponownie wcześniej
  emulowanych shaderów; obejście dla AMD na Windowsie przebudowuje je w trybie natywnym. Użyj `--wave64-emulate on`,
  aby wymusić emulację, albo `--wave64-emulate off`, aby wymusić natywne wave64.
- **Cztery flagi awaryjne** są przekazywane do translatora w ten sam sposób, zapisywane obok przetłumaczonych
  shaderów, a ich zmiana tłumaczy je ponownie: `--no-float-controls` (usuwa `SignedZeroInfNanPreserve` dla urządzeń,
  których sprawdzanie potoku pomija shader bez `shaderSignedZeroInfNanPreserveFloat32`; wymaga flagi translatora z
  PR #664), `--bary-fallback` (czyta surowe barycentryki z interpolowanego wejścia, dla urządzeń bez
  `VK_KHR_fragment_shader_barycentric`), `--texel-offset-fallback` (wlicza przesunięcia `image_sample*_o` do
  współrzędnej, dla urządzeń bez `VK_KHR_maintenance8`) i `--mesh-compute` (shadery mesh jako compute, dla urządzeń
  bez `VK_EXT_mesh_shader`). Każda jest opisana w [`docs/design.md`](../../design.md).
- **`--work`**: najlepiej wskazać dysk z co najmniej 400 GB wolnego miejsca; do samego uruchomienia wystarczy około
  30 GB (zobacz „Mało miejsca na dysku” powyżej).
- **`--wav`**: domyślnie wyłączone, bo 16-bitowy PCM WAV jest kilkakrotnie większy od Ogg Opus. Dekoduje przepakowane
  strumienie Ogg Opus do `work\assets\audio\wav` po etapie banków audio; istniejące pliki WAV są pomijane przy
  wznawianiu builda.
- **Linux i Wine/Proton**: `make_game.py` działa też na Linuksie, a etap `pipelines` sam buduje tam
  `vk_pipeline_check`, jeśli go brakuje. Aby wykonać etap `runtime` poza Windowsem, podaj `--runtime-build <folder>`
  z buildem dla Windowsa (`gta6rt.exe` i `gta6rt_tls_host.dll`, np. rozpakowany artefakt `runtime-windows` z CI).
  Skrypt skopiuje te pliki do `--out` i utworzy `gta6rt.sh`, który uruchamia program przez `wine` (albo przez Wine
  wskazane zmienną `WINE`, np. z Protona: `files/bin/wine`). Ta sama opcja pomija build na Windowsie. Alternatywnie
  etap `runtime` może na Linuksie cross-kompilować loader przez clang-cl i lld-link (LLVM 19+, ale nie 22.1.7 ani
  22.1.8) przeciw CRT MSVC i Windows SDK przygotowanym przez [xwin](https://github.com/Jake-Shadle/xwin) w
  `XWIN_DIR` lub `~/.local/share/xwin-sdk`; szczegóły w
  [`docs/testing.md`](../../testing.md#the-windows-runtime-on-linux-under-wine-or-proton).

Build zapisuje polecenie i wersję `llvm-mc`, które utworzyły pliki `.s`, oraz źródła translatora, które utworzyły
shadery. Zmiana któregokolwiek z nich, a także użycie starszego folderu roboczego, uruchamia ponownie deasemblację
albo tłumaczenie z zachowaniem wyodrębnionych wejść shaderów.
- **`--wave64-emulate`**: emuluje shadery compute/mesh wave64 na kartach z podgrupami 32-wątkowymi (NVIDIA i Intel). Domyślny tryb `auto` pyta `vulkaninfo` i sam włącza emulację, gdy jest potrzebna.
- **`--work`**: najlepiej wskazać dysk z co najmniej 400 GB wolnego miejsca. Etap `shaders` trwa najdłużej i domyślnie używa połowy wątków procesora (można to zmienić przez `--jobs`).
- **Linux i Wine/Proton**: `make_game.py` działa też na Linuksie. Aby wykonać etap `runtime` poza Windowsem, podaj `--runtime-build <folder>` z buildem dla Windowsa (`gta6rt.exe` i `gta6rt_tls_host.dll`). Skrypt utworzy wtedy `gta6rt.sh`, który uruchamia program przez `wine` (albo przez Wine wskazane zmienną `WINE`, np. z Protona).

## Częste problemy

- **Pobrałeś repozytorium przyciskiem „Download ZIP” na GitHubie**: archiwum ZIP nie zawiera submodułu
  `third_party/rpf9-explorer`. Użyj `git clone --recursive`; build pobierze submoduł sam, jeśli masz `git`, a jeśli
  nie — powie, co zrobić. Możesz też uruchomić `git submodule update --init --recursive` w folderze repozytorium.
- **`llvm-mc` odrzuca `gfx1013` / AMDGPU**: najpewniej masz oficjalny instalator LLVM dla Windowsa. Użyj
  `scripts\build_llvm_mc.ps1` albo ustaw `LLVM_MC` na `llvm-mc` z WSL (`llvm-21`).
- **Błąd `D3D12_ROOT_SIGNATURE_DESC2` przy budowaniu runtime**: zaktualizuj Windows SDK do `10.0.26100` lub nowszego
  przez Visual Studio Installer albo `winget install Microsoft.WindowsSDK.10.0.26100`, usuń folder builda i
  skonfiguruj go ponownie.
- **Brak miejsca na dysku `C:`**: podaj `--work "D:\leonida-work"`, aby przenieść pliki robocze na inną partycję.
  Do samego uruchomienia wystarczą etapy `--stages check,shaders,runtime` (około 30 GB); `--out` też może leżeć na
  innym dysku.

## Jak pomóc i uruchomić testy GPU

1. Przejrzyj otwarte PR-y i [issue koordynacyjne][coord], żeby nie dublować pracy, i zarezerwuj zadanie **przed**
   otwarciem PR-a: wpisem `claiming: <ID>` albo jednym zdaniem opisu, jeśli zadanie nie ma jeszcze ID.
2. Uruchom odpowiednie testy na syntetycznych danych wejściowych. Na przykład weryfikator manifestów testujesz
   poleceniem `python -I tests/test_manifest_verifier.py` w głównym folderze repozytorium, a suite translatora —
   `python -m unittest discover -s tests/semantic -v`.
3. W PR podaj ID zadania, opisz, co przetestowałeś, a jeśli uruchamiałeś testy GPU — podaj kartę i wersję
   sterownika. Zaznacz, czy korzystałeś z pomocy AI, zgodnie z [`AGENTS.md`](../../../AGENTS.md). Zasady znajdziesz
   w [`CONTRIBUTING.md`](../../../CONTRIBUTING.md).

**Nigdy** nie załączaj do issues ani PR-ów plików gry, wyodrębnionych zasobów ani wygenerowanych shaderów: dziel się
tylko wynikami (liczby, wyniki testów, klasy błędów).

[Przewodnik po testach GPU](../../testing.md) pokazuje, jak uruchomić testy bez plików gry. Wyniki testów GPU
publikuj w [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166).

[coord]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22
