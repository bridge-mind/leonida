# Быстрый старт Leonida (`ru`)

> [!NOTE]
> Главный источник: английская документация ([`README.md`](../../../README.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)
> и [`docs/README.md`](../../README.md)). Этот гайд покрывает только стабильные шаги установки и сборки
> (сверен с коммитом `aa2988cc`). Текущее состояние проекта меняется каждый день, его смотрите в
> [`README.md`](../../../README.md).

## Что такое Leonida

Leonida это проект статической рекомпиляции Grand Theft Auto VI (Xbox Series): игра запускается нативно на
Windows. Процессор консоли x86-64, как у ПК, поэтому код игры загружается и работает напрямую, а системное ПО
консоли реализовано заново. Готовые GPU-программы RDNA2 переводятся в SPIR-V для Vulkan. Модели, текстуры и звук
читаются из вашего пакета. Всё это происходит **на вашем ПК, из вашей копии игры**.

> [!IMPORTANT]
> В репозитории **нет ни игры, ни ключей, ни исполняемого файла**, и он не объясняет, где их взять. Нужна своя
> копия, файлы которой уже читаются с диска. Ничто в проекте не расшифровывает данные и не обходит защиты. Всё,
> что создаёт сборка (переведённые шейдеры, извлечённые ресурсы, рабочая папка и ваши сохранения), берётся из
> вашей игры и **остаётся на вашем ПК**. Не выкладывайте эти файлы (см. [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)).

Проект на раннем этапе: **игра пока не запускается**.

## Платформы

Ни на одной платформе пока ничего не играбельно. Что где работает сегодня:

- **Windows**: всё (`make_game.py`, инструменты и runtime `gta6rt` со слоем D3D12 → Vulkan и его тестами).
- **Linux**: переводчик, инструменты ресурсов и инструменты GPU-тестов Vulkan собираются и работают нативно
  (проверено на RADV и NVIDIA). Runtime это код для Windows: он кросс-собирается через clang-cl и xwin, а его тесты
  идут под Wine или Proton. Игра должна запускаться так же (#217).
- **macOS (Apple Silicon)**: инструменты на Python и инструменты тестов Vulkan работают нативно, на MoltenVK
  (Homebrew) или KosmicKrisp. Runtime идёт под Wine, как на Linux. Известные ограничения: нет этапа geometry и
  `VK_EXT_mesh_shader`, подгруппы шириной 32 (шейдеры wave64 эмулируются) и работа MoltenVK с числами с плавающей
  точкой; все они перечислены в [`docs/testing.md`](../../testing.md).
- **AMD RDNA1 (RX 5000)**: поддерживаемая цель, близкая к RDNA2 консоли; её подгруппы шириной 64 выполняют
  шейдеры wave64 нативно.

Результаты GPU со всех платформ собираются в [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166).

## Что нужно

- **Своя копия** пакета Xbox Series версии `1.0.2.1`: папка, где `MicrosoftGame.Config` и архивы `.rpf` лежат
  обычными файлами.
- **Windows 10/11 x64** и примерно **350 ГБ** свободного места под рабочие файлы. Их можно перенести на другой
  диск через `--work`.
- [Python 3.12+](https://www.python.org/downloads/) (при установке отметьте *«Add python.exe to PATH»*).
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).
- [Vulkan SDK](https://vulkan.lunarg.com/) для `spirv-as` и `spirv-val`.
- [LLVM 21+](https://github.com/llvm/llvm-project/releases) с **целью AMDGPU** для `llvm-mc`:
  - в официальной сборке LLVM для Windows цели AMDGPU нет;
  - на Windows `scripts\build_llvm_mc.ps1` собирает свой `llvm-mc.exe` с этой целью (без WSL, нужны `git` и
    инструменты C++ из Visual Studio ниже) и прописывает `LLVM_MC`; после этого откройте новый терминал, чтобы
    переменная подхватилась. Сборка занимает 20-40 минут;
  - быстрее, если вы пользуетесь [conda](https://docs.conda.io/): `conda install -c conda-forge llvm-tools` (в её
    сборке для Windows цель AMDGPU есть), затем укажите в `LLVM_MC` полный путь к установленному `llvm-mc.exe`
    (`where.exe llvm-mc` покажет его в активированном окружении; в PowerShell команда `where` без `.exe` — это
    `Where-Object`);
  - если ставите `llvm-21` в WSL, передайте `LLVM_MC` целой командой в той же сессии PowerShell, откуда
    запускаете `make_game.py`:

    ```powershell
    $env:LLVM_MC = 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'
    ```

    Подробности в [`docs/design.md`](../../design.md#disassembling-shaders);
  - на Linux достаточно `llvm-mc` в `PATH`.
- **Visual Studio 2022** с рабочей нагрузкой *«Разработка классических приложений на C++»* и **Windows SDK
  `10.0.26100` или новее**: они нужны этапу `runtime`, который использует root signatures версии 1.2, а в старых
  SDK их нет.
- [ffmpeg](https://ffmpeg.org/download.html) в `PATH` *(по желанию: декодирует потоки XMA2, обязателен для `--wav`)*.

## Как собрать

### Простой способ (Windows)

1. Установите всё из списка выше.
2. Склонируйте проект **вместе с подмодулем**:
   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```
   Кнопка «Download ZIP» на GitHub оставляет за бортом `third_party/rpf9-explorer`. Сборка подтянет его сама,
   если установлен `git`, иначе скажет, что сделать.
3. Запустите **`Build.bat`** двойным щелчком.

Скрипт сам находит копию игры: смотрит папки «Загрузки», рабочий стол, «Документы» (в том числе когда их хранит
OneDrive), `Games`, домашнюю папку и корень каждого диска, на два уровня вглубь. Версию он определяет по
размерам нескольких файлов, не читая больших данных. Если копий несколько, спросит, какую брать;
если ни одной, попросит вставить путь к папке. Перед стартом он проверит, что всё из списка установлено и места
хватает, и покажет все недостающие пункты разом. Результат пишется в `Documents\Leonida`. Прерванная
сборка продолжается с того места, где встала.

### Мало места на диске

Объём, указанный выше, нужен для полной сборки. Чтобы *запустить* порт, `--stages check,shaders,runtime` хватает
примерно 30 ГБ; этап `assets`, на который приходится почти всё остальное, нужен только инструментам ресурсов, а
`--wav` добавляет ещё в несколько раз больше, чем занимает Ogg Opus. `--work` и `--out` можно вынести на другой диск. Runtime читает пакет на месте, поэтому для
диска только для чтения нужен `gta6rt --overlay <папка с правом записи>` (это флаг `gta6rt`, а не `make_game.py`).
Не удаляйте файлы `.s` в `work/shaders/dxbc/disasm`: без них и без `work/shaders/spirv/index.json` этап `shaders`
может выглядеть завершённым с нулём шейдеров.

### Из терминала (`make_game.py`)

```powershell
python -I make_game.py --find                          # какие копии нашлись на этом ПК
python -I make_game.py --pick                          # выбрать папку игры в диалоге
python -I make_game.py --check                         # проверить копию, ничего не собирать
python -I make_game.py --check --full-check            # ещё и сравнить SHA-256 каждого файла (долго)
python -I make_game.py --game "D:\путь\к\пакету" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets         # только часть этапов
python -I make_game.py --game "D:\путь\к\пакету" --wav # ещё и Opus в PCM WAV (много места)
python -I make_game.py --wave64-emulate off            # auto (по умолчанию), on или off
```

Этапы идут в таком порядке: `check`, `shaders`, `assets`, `runtime`, `pipelines`. Последний создаёт все
переведённые шейдеры на драйвере вашей видеокарты, проверяет, что они компилируются, сообщает о пропущенных и
сохраняет кеш конвейеров в `pipeline_cache.bin` в `--out`. Runtime этот кеш пока не загружает, так что первую
загрузку он не ускоряет. Перевод шейдеров самый долгий: он выполняется в `--jobs` процессов (по умолчанию
половина потоков процессора).

- **`--wave64-emulate`**: переводить ли compute- и mesh-шейдеры wave64 так, чтобы они работали точно при любом
  размере подгруппы (раздел «Wave64 emulation» в [`docs/design.md`](../../design.md)). У NVIDIA и Intel подгруппы
  шириной 32, им это нужно. Режим `auto` спрашивает `vulkaninfo` из Vulkan SDK и включает эмуляцию в трёх
  случаях: не все видеокарты сообщают подгруппы шириной 64, ответ получить не удалось, или карта для runtime не
  умеет выполнять compute- и mesh-шейдеры на 64 (с `--mesh-compute` нужна только поддержка compute). На драйвере
  AMD для Windows `auto` берёт нативный wave64 на RDNA1 и RDNA2, если карта для runtime его поддерживает, даже
  когда другая карта шириной 32 или `vulkaninfo` недоступен: этот драйвер может падать на эмулированном коде LDS
  (#857). Режим сохраняется рядом с переведёнными шейдерами, и его смена запускает перевод заново. При `auto`
  обычно можно переиспользовать уже эмулированные шейдеры; обход для драйвера AMD пересобирает их в нативном
  режиме.
- **`--work`**: выберите диск, где свободно хотя бы 350 ГБ.
- **`--wav`**: по умолчанию выключен, потому что 16-битный PCM WAV в несколько раз больше Ogg Opus. Флаг
  декодирует перепакованные потоки Ogg Opus в `work\assets\audio\wav` после этапа аудио; готовые WAV при
  продолжении сборки пропускаются.

Четыре запасных флага передаются переводчику так же, сохраняются рядом с шейдерами и запускают перевод заново
при изменении: `--no-float-controls` (убрать `SignedZeroInfNanPreserve` для устройств, чья проверка конвейеров
пропускает шейдер без `shaderSignedZeroInfNanPreserveFloat32`), `--bary-fallback` (читать сырые барицентрики из
интерполированного входа, для устройств без `VK_KHR_fragment_shader_barycentric`), `--texel-offset-fallback`
(вносить смещения `image_sample*_o` в координату, для устройств без `VK_KHR_maintenance8`) и `--mesh-compute`
(mesh-шейдеры как compute, для устройств без `VK_EXT_mesh_shader`). Каждый описан в
[`docs/design.md`](../../design.md).

Сборка запоминает, чем сделаны результаты: команду и версию `llvm-mc`, исходники переводчика и декодер ресурсов
(коммит `third_party/rpf9-explorer` и исходники `tools/rpf9dump`). Если что-то из этого меняется, заново
выполняется только нужная часть: дизассемблирование, перевод или экспорт моделей и текстур архива. Старые рабочие
папки без такой записи один раз проходят эти шаги повторно после обновления.

### Linux и Wine/Proton

`make_game.py` работает и на Linux. Этап `pipelines` использует `vk_pipeline_check` (CMake 3.24+, компилятор
C++20 и заголовки Vulkan; собирается в `runtime/build/linux`, а если его нет, скрипт соберёт его сам до остальной
работы). Загрузчик (`gta6rt.exe`) это программа для Windows, поэтому на Linux она
идёт под Wine или Proton. Этап `runtime` на Linux кросс-собирается в `runtime/build/windows` через `clang-cl` и
`lld-link`
(LLVM 19+, но не 22.1.7 и не 22.1.8, см. [`docs/testing.md`](../../testing.md#known-problems)) против CRT от MSVC и
Windows SDK, которые раскладывает [xwin](https://github.com/Jake-Shadle/xwin) в `XWIN_DIR` или
`~/.local/share/xwin-sdk`. Разовый вызов `xwin` есть в
[`docs/testing.md`](../../testing.md#the-windows-runtime-on-linux-under-wine-or-proton). Если этих инструментов нет,
этап выпадает из списка по умолчанию, а при явном запросе скрипт сразу перечислит, чего не хватает. Готовая
сборка для Windows тоже подойдёт, например артефакт `runtime-windows` из CI:

```sh
python3 -I make_game.py --runtime-build ~/Downloads/runtime-windows
```

В обоих случаях `gta6rt.exe` и `gta6rt_tls_host.dll` копируются в `--out`, а рядом пишется `gta6rt.sh`, который
запускает `gta6rt.exe` через `wine` или через Wine из переменной `WINE` (для Proton это его `files/bin/wine`). На
Windows этот же флаг пропускает сборку.

## Частые проблемы

- **Скачали ZIP с GitHub**: в архиве нет подмодуля `third_party/rpf9-explorer`. Используйте
  `git clone --recursive` или выполните `git submodule update --init --recursive` в папке репозитория.
- **`llvm-mc` не знает `gfx1013` или цель AMDGPU**: скорее всего установлен официальный LLVM для Windows.
  Соберите свой через `scripts\build_llvm_mc.ps1` или укажите в `LLVM_MC` путь к `llvm-mc` из WSL (`llvm-21`).
- **CMake пишет «needs the Windows SDK 10.0.26100 or newer»** или MSVC не находит `Desc_1_2` в
  `D3D12_VERSIONED_ROOT_SIGNATURE_DESC`: поставьте Windows SDK `10.0.26100` или новее через Visual Studio
  Installer либо `winget install Microsoft.WindowsSDK.10.0.26100`, удалите `runtime/build` и настройте заново.
- **Кончилось место на `C:`**: передайте `--work "D:\leonida-work"`, чтобы рабочие файлы легли на другой диск.
- **Windows не даёт запустить только что собранные тестовые программы** («An Application Control policy has
  blocked this file»): так работает Smart App Control в Windows 11, подробности в
  [`docs/testing.md`](../../testing.md#known-problems).

## Как помочь и прогнать тесты на видеокарте

1. Посмотрите открытые pull request и [issue координации][coord], чтобы не делать ту же работу дважды. Заявите
   задачу строкой `claiming: <ID>` или одним предложением описания, если ID ещё нет.
2. Прогоните нужные тесты на синтетических входных данных. Например, проверку манифеста:
   `python -I tests/test_manifest_verifier.py` из корня репозитория.
3. В pull request укажите ID задачи и то, что вы проверили, а если запускали тесты на видеокарте, ещё её модель и
   версию драйвера. Отметьте, помогал ли вам ИИ. Все правила в [`CONTRIBUTING.md`](../../../CONTRIBUTING.md).

[Руководство по тестам на видеокарте](../../testing.md) объясняет, как обойтись без файлов игры. Результаты
складывают в [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166), по одному комментарию на карту и
драйвер.

[coord]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22
