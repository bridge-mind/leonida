# Leonida — Hızlı Başlangıç (`tr`)

> [!NOTE]
> İngilizce dokümantasyon ([`README.md`](../../../README.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) ve
> [`docs/README.md`](../../README.md)) **yetkili kaynaktır**. Bu rehber yalnızca kararlı kurulum ve derleme
> yönergelerini kapsar (`561b65a` commit'ine göre eşitlenmiştir). Günlük değişen proje durumu için
> [`README.md`](../../../README.md) dosyasına bakın.

## Leonida Nedir?

Leonida, Grand Theft Auto VI'nın Xbox Series sürümünü Windows üzerinde yerel olarak çalıştırmayı hedefleyen statik
yeniden derleme (recompilation) projesidir. Port, **kendi bilgisayarınızda ve kendi oyun kopyanızdan** oluşturulur.
Konsolun önceden derlenmiş RDNA2 GPU shader'ları Vulkan için SPIR-V'ye çevrilir, oyun varlıkları doğrudan kendi
paketinizden okunur ve konsolun sistem yazılımları shim katmanlarıyla yeniden uygulanır.

> [!IMPORTANT]
> Bu depo **oyunu, şifre çözme anahtarlarını veya çalıştırılabilir dosyayı içermez** ve bunların nasıl edinileceğini
> anlatmaz. Dosyaları diskte zaten okunabilir durumda olan kendi kopyanız gerekir. Buradaki hiçbir araç şifre çözmez
> veya herhangi bir korumayı aşmaz. Derlemenin ürettiği tüm dosyalar (çevrilmiş shader'lar, çıkarılan varlıklar,
> çalışma klasörü ve kayıtlar) kendi kopyanızdan gelir ve **yalnızca bilgisayarınızda kalmalıdır**; bunları asla issue
> veya pull request'lerde paylaşmayın (bkz. [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)).

Proje henüz erken aşamadadır: **oyun şu an açılmıyor**.

## Gereksinimler

- **Kendi Xbox Series paket kopyanız**, sürüm `1.0.2.1` (`MicrosoftGame.Config` dosyasını ve `.rpf` arşivlerini diskte
  düz dosyalar olarak barındıran klasör).
- **Windows 10/11 x64** ve çalışma dosyaları için yaklaşık **400 GB boş disk alanı** (`--work` ile başka bir diske
  yönlendirilebilir). Bu rakam tam derleme içindir: yalnızca çalıştırmak için `--stages check,shaders,runtime`
  yaklaşık 30 GB ister, `assets` aşaması ise yaklaşık 370 GB.
- [Python 3.12+](https://www.python.org/downloads/) (kurulumda *"Add python.exe to PATH"* seçeneğini işaretleyin).
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).
- [Vulkan SDK](https://vulkan.lunarg.com/) (`spirv-as`, `spirv-val` ve `vulkaninfo` için).
- **AMDGPU hedefli** [LLVM 21+](https://github.com/llvm/llvm-project/releases) (`llvm-mc`):
  - Resmi Windows LLVM kurulumu AMDGPU hedefini **içermez**.
  - AMDGPU destekli yerel `llvm-mc.exe` derlemek için depo kök dizininde PowerShell'den `.\scripts\build_llvm_mc.ps1`
    betiğini çalıştırın (`git` ve aşağıdaki Visual Studio C++ araçları gerekir). Betik kullanıcınız için `LLVM_MC`
    ortam değişkenini ayarlar; geçerli olması için yeni bir terminal açın.
  - Daha hızlı bir yol: [conda](https://docs.conda.io/) kullanıyorsanız `conda install -c conda-forge llvm-tools`
    (Windows derlemesi AMDGPU hedefini içerir; [`docs/testing.md`](../../testing.md) 23.1.6 sürümünü kullandı), ardından
    `LLVM_MC` değişkenini kurduğu `llvm-mc.exe` dosyasının tam yoluna ayarlayın (ortam etkinleştirildikten sonra
    `where.exe llvm-mc` yolunu gösterir; PowerShell'de çıplak `where` komutu `Where-Object` demektir).
  - Alternatif olarak WSL (Ubuntu) içinde `llvm-21` kurup `make_game.py` çalıştıracağınız PowerShell oturumunda
    `LLVM_MC` değişkenini tam komut olarak ayarlayın:

    ```powershell
    $env:LLVM_MC = 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'
    ```

    Ayrıntılar için [`docs/design.md`](../../design.md#disassembling-shaders) bölümüne bakın.
  - Linux üzerinde `PATH` içinde `llvm-mc` bulunması yeterlidir.
- **Visual Studio 2022** (*C++ ile masaüstü geliştirme* iş yükü) ve **Windows SDK `10.0.26100` veya üzeri** (`runtime`
  aşaması için; sürüm 1.2 root signature yapıları daha eski SDK'larda yoktur).
- `PATH` içinde [ffmpeg](https://ffmpeg.org/download.html) *(isteğe bağlı: XMA2 ses akışlarını çözer; `--wav` için
  gereklidir)*.

## Nasıl Derlenir?

### Kolay yöntem (Windows)

1. Yukarıdaki gereksinimleri kurun.
2. Depoyu **alt modülüyle birlikte** klonlayın:
   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```
3. **`Build.bat`** dosyasına çift tıklayın.

Betik oyun kopyanızı kendisi arar (İndirilenler, Masaüstü, Belgeler, Games, kullanıcı klasörünüz ve her sürücünün kök
dizininden iki seviye derine kadar). Büyük dosyaları okumadan yalnızca birkaç dosya boyutundan sürümü tanır. Birden
fazla kopya bulursa hangisini kullanacağınızı, hiç bulamazsa klasör yolunu yapıştırmanızı ister. Başlamadan önce
gereksinimleri ve boş disk alanını denetleyip eksikleri tek seferde listeler. Çıktı `Documents\Leonida` klasörüne
yazılır; yarım kalan bir derleme kaldığı yerden devam eder.

### Terminalden (`make_game.py`)

```powershell
python -I make_game.py --find                          # bilgisayarda bulunan kopyaları listele
python -I make_game.py --pick                          # oyun klasörünü bir klasör penceresinde seç
python -I make_game.py --check                         # kopyayı doğrula, derleme yapma
python -I make_game.py --check --full-check            # ayrıca her dosyanın SHA-256 özetini karşılaştır (yavaş)
python -I make_game.py --game "D:\paket\yolu" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets         # yalnızca seçili aşamaları çalıştır
python -I make_game.py --game "D:\paket\yolu" --wav    # Opus akışlarını ayrıca PCM WAV'a çöz (büyük boyut)
python -I make_game.py --wave64-emulate off            # auto (varsayılan), on veya off
```

Aşamalar şu sırayla çalışır: `check` → `shaders` → `assets` → `runtime` → `pipelines`.

- **`--wave64-emulate`**: Wave64 compute ve mesh shader'larını her subgroup genişliğinde birebir çalışacak şekilde
  çevirir. NVIDIA ve Intel GPU'lar 32 şeritli subgroup kullandığı için buna ihtiyaç duyar. Varsayılan `auto` modu
  `vulkaninfo` çıktısına bakar ve tüm GPU'lar yerel 64 şeritli subgroup bildirmediği sürece emülasyonu otomatik açar.
- **`--work`**: Tam derleme için en az 400 GB boş alana sahip bir sürücü seçin (yalnızca çalıştırmak için yaklaşık
  30 GB, `assets` aşaması için yaklaşık 370 GB yeterlidir). En uzun adım shader çevirisidir ve varsayılan olarak
  CPU iş parçacıklarının yarısını kullanır (`--jobs` ile değiştirilebilir).
- **`--wav`**: Varsayılan olarak kapalıdır; 16-bit PCM WAV dosyaları Ogg Opus'tan birkaç kat daha büyüktür.
- **Linux ve Wine/Proton**: `make_game.py` Linux'ta da çalışır. Yükleyici (`gta6rt.exe`) bir Windows programıdır ve
  Wine veya Proton altında çalışır. Linux'ta `runtime` aşaması, `clang-cl` + `lld-link` (LLVM 19+) ve
  [xwin](https://github.com/Jake-Shadle/xwin) (`XWIN_DIR` veya `~/.local/share/xwin-sdk`; tek seferlik kurulum için
  bkz. [`docs/testing.md`](../../testing.md#the-windows-runtime-on-linux-under-wine-or-proton)) ile `runtime/build/windows`
  içine çapraz derleme yapabilir ya da `--runtime-build <klasör>` ile önceden derlenmiş bir Windows runtime'ını
  (`gta6rt.exe` ve `gta6rt_tls_host.dll`, örn. CI'daki `runtime-windows` paketi) doğrudan kopyalayabilir. Her iki durumda
  da çıkış klasörüne `gta6rt.sh` başlatıcısını yazar (`wine` veya `WINE` ortam değişkeni ile çalıştırılır).

## Sık Karşılaşılan Sorunlar

- **GitHub'dan "Download ZIP" ile indirdiniz**: ZIP arşivi `third_party/rpf9-explorer` alt modülünü içermez.
  `git clone --recursive` kullanın veya depo klasöründe `git submodule update --init --recursive` çalıştırın.
- **`llvm-mc`, `gfx1013` veya AMDGPU hedefini reddediyor**: Büyük olasılıkla resmi Windows LLVM kurulumu kullanılıyor.
  `scripts\build_llvm_mc.ps1` ile yerel derleyin veya `LLVM_MC` değişkenini WSL içindeki `llvm-21` `llvm-mc` yoluna
  ayarlayın.
- **CMake "needs the Windows SDK 10.0.26100 or newer" hatası veriyor** (veya MSVC `Desc_1_2` üyesini bulamıyor):
  Visual Studio Installer üzerinden ya da `winget install Microsoft.WindowsSDK.10.0.26100` komutuyla Windows SDK'yı
  güncelleyin, `runtime/build` klasörünü silip yeniden yapılandırın.
- **`C:` sürücüsünde yer kalmadı**: Çalışma dosyalarını başka bir sürücüye taşımak için `--work "D:\leonida-work"`
  parametresini verin.
- **Windows yeni derlenen test programlarını engelliyor** (*"An Application Control policy has blocked this file"*):
  Windows 11 Akıllı Uygulama Denetimi kaynaklıdır; bkz. [`docs/testing.md`](../../testing.md#known-problems).

## Katkıda Bulunma ve GPU Testleri

1. Aynı işi iki kez yapmamak için açık PR'lara ve [koordinasyon issue'suna][coord] göz atın. Başlamadan önce
   `claiming: <ID>` yazarak (veya henüz ID yoksa tek satırlık açıklamayla) görevi sahiplenin.
2. Sentetik girdilerle ilgili testleri çalıştırın (örneğin depo kökünde `python -I tests/test_manifest_verifier.py`).
3. PR açıklamasında görev ID'sini, neleri test ettiğinizi ve GPU testi çalıştırdıysanız ekran kartı ile sürücü
   sürümünü belirtin; yapay zekâ desteği kullandıysanız not düşün. Tüm kurallar için
   [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) dosyasına bakın.

[GPU test rehberi](../../testing.md), oyun dosyası olmadan testlerin nasıl çalıştırılacağını açıklar. GPU test
sonuçlarını [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166) altında (GPU ve sürücü başına bir yorum)
paylaşabilirsiniz.

[coord]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22
