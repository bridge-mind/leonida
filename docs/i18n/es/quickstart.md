# Leonida — Guía de inicio (`es`)

> [!NOTE]
> La documentación en inglés ([`README.md`](../../../README.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) y
> [`docs/README.md`](../../README.md)) es **la referencia oficial**. Esta guía solo recoge las instrucciones
> estables de instalación y compilación (sincronizada con el commit `aa2988cc`). Para el estado del proyecto, que
> cambia a diario, consulte el [`README.md`](../../../README.md).

## Qué es Leonida

Leonida es un proyecto de recompilación estática de Grand Theft Auto VI (versión Xbox Series) para ejecutarlo de
forma nativa en Windows. Se compila **en su PC, a partir de su propia copia del juego**. Los shaders RDNA2 de la
consola se traducen a SPIR-V (Vulkan), los recursos se leen directamente de su paquete y el software de sistema de
la consola se reimplementa como shims.

> [!IMPORTANT]
> Este repositorio **no contiene el juego, ni claves, ni ejecutables**, y no explica cómo conseguirlos. Nada de lo
> que hay aquí descifra ni elude una protección. Todo lo que la compilación produce (shaders traducidos, recursos
> extraídos, carpeta de trabajo, guardados) proviene de su copia y **debe permanecer en su máquina**: no adjunte ni
> comparta nunca esos archivos en una issue o en una pull request (véase
> [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)).

El proyecto está en una fase inicial: **el juego todavía no arranca**.

## Requisitos previos

- **Su propia copia** del paquete de Xbox Series, versión `1.0.2.1`: la carpeta que contiene `MicrosoftGame.Config`
  y los archivos `.rpf`, como ficheros legibles en el disco.
- **Windows 10/11 x64**, con unos **400 GB libres** para los archivos de trabajo. Pueden ir a otra unidad con
  `--work`; más abajo se explica cómo omitir la exportación de recursos y usar unos 30 GB.
- [Python 3.12+](https://www.python.org/downloads/) (marque «Add python.exe to PATH» durante la instalación).
- [SDK de .NET 8](https://dotnet.microsoft.com/download/dotnet/8.0).
- [SDK de Vulkan](https://vulkan.lunarg.com/) (`spirv-as`, `spirv-val` y `vulkaninfo`).
- [LLVM 21+](https://github.com/llvm/llvm-project/releases) con el **destino AMDGPU** (`llvm-mc`):
  - La versión oficial de LLVM para Windows **no incluye** el destino AMDGPU.
  - Para compilar un `llvm-mc.exe` nativo con ese destino, ejecute `.\scripts\build_llvm_mc.ps1` en PowerShell
    desde la raíz del repositorio. El script necesita `git` y las herramientas de C++ de Visual Studio. Define
    `LLVM_MC` para su usuario: abra después un terminal nuevo para que la variable se aplique.
  - Si usa [conda](https://docs.conda.io/), `conda install -c conda-forge llvm-tools` ofrece una alternativa cuyo
    paquete para Windows incluye AMDGPU. Con el entorno activado, `where.exe llvm-mc` muestra la ruta del
    ejecutable; asigne esa ruta completa a `LLVM_MC`.
  - Con WSL, instale `llvm-21` en Ubuntu y pase el comando completo a `LLVM_MC`, en la sesión de PowerShell
    donde vaya a ejecutar `make_game.py`:

    ```powershell
    $env:LLVM_MC = 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'
    ```

    Véase [`docs/design.md`](../../design.md#disassembling-shaders) para los detalles.
  - En Linux basta con un `llvm-mc` presente en el `PATH`.
- **Visual Studio 2022** con la carga de trabajo *Desarrollo para el escritorio con C++* y el **SDK de Windows
  `10.0.26100` o superior**, para la etapa `runtime` (el runtime usa las root signatures 1.2, ausentes en SDK
  más antiguos).
- [ffmpeg](https://ffmpeg.org/download.html) en el `PATH` *(opcional: decodifica los flujos XMA2 y es obligatorio
  con `--wav`)*.

## Compilar

### La forma sencilla (Windows)

1. Instale los requisitos anteriores.
2. Clone el repositorio **con su submódulo**:
   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```
3. Haga doble clic en **`Build.bat`**.

El script encuentra su copia por sí solo: busca en Descargas, Escritorio y Documentos (también si Windows los ha
movido a OneDrive), Games, su carpeta de usuario y en la raíz de cada disco, dos niveles de carpetas en
profundidad. Reconoce la versión por el tamaño de algunos archivos, sin leer archivos grandes. Si encuentra
varias copias, le pregunta cuál usar; si no encuentra ninguna, le pide que pegue la ruta de la carpeta. Antes de
empezar comprueba los requisitos y el espacio libre, y enumera de una vez todo lo que falta. El resultado va a
`Documents\Leonida`. Una compilación interrumpida se reanuda donde quedó.

### Si dispone de poco espacio

Los 400 GB corresponden a la compilación completa. Puede omitir la exportación de recursos con
`--stages check,shaders,runtime`, que necesita unos **30 GB**. La etapa `assets` (unos 370 GB) sirve para las
herramientas de recursos; `--wav` añade varias veces el tamaño de los archivos Ogg Opus. La comprobación previa
ofrece la alternativa sin `assets` cuando esa etapa no cabe. Puede situar tanto `--work` como `--out` en otra
unidad.

El runtime accede al paquete en su ubicación original. Si la unidad del paquete es de solo lectura, use
`gta6rt --overlay <carpeta con permisos de escritura>`: `--overlay` es una opción de `gta6rt`.

Conserve los archivos `.s` de `work/shaders/dxbc/disasm`: borrarlos junto con
`work/shaders/spirv/index.json` puede hacer que la etapa de shaders parezca terminada con cero shaders. Si la
etapa necesita vaciar `disasm`, rechazará esa carpeta cuando sea un enlace simbólico.

### Desde un terminal (`make_game.py`)

```powershell
python -I make_game.py --find                          # enumera las copias encontradas en este PC
python -I make_game.py --pick                          # elige la carpeta mediante un diálogo
python -I make_game.py --check                         # verifica su copia, sin compilar nada
python -I make_game.py --check --full-check            # también comprueba los SHA-256 de todos los archivos (lento)
python -I make_game.py --game "D:\ruta\al\paquete" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages check,shaders,runtime   # sin la exportación de recursos (unos 30 GB)
python -I make_game.py --stages shaders,assets         # solo algunas etapas
python -I make_game.py --game "D:\ruta\al\paquete" --wav  # además decodifica el Opus a WAV PCM (ocupa mucho)
python -I make_game.py --wave64-emulate off            # auto (predeterminado), on u off
```

Las etapas se ejecutan en este orden: `check` → `shaders` → `assets` → `runtime` → `pipelines`.

La traducción de shaders es la etapa más larga; por defecto usa la mitad de los hilos del procesador (ajustable
con `--jobs`). La etapa `pipelines` comprueba que el controlador de su GPU puede crear los shaders traducidos e
informa de los que debe omitir. Guarda `pipeline_cache.bin` en `--out`, pero el runtime todavía no carga esa
caché, así que no acelera el primer arranque.

`--wav` está desactivado por defecto, porque el WAV PCM de 16 bits ocupa varias veces más que el Ogg Opus.
Cuando se activa, decodifica los Ogg Opus a `work/assets/audio/wav` después de exportar los bancos de audio;
al reanudar, omite los WAV ya existentes.

### Emulación wave64 y compatibilidad de la GPU

`--wave64-emulate` decide si los shaders compute y mesh wave64 se traducen de forma exacta para cualquier tamaño
de subgrupo. Las GPU NVIDIA e Intel usan subgrupos de 32 y lo necesitan. El valor predeterminado, `auto`, consulta
`vulkaninfo` y activa la emulación si alguna GPU no admite subgrupos de 64 o su tamaño es desconocido. También
comprueba que la GPU del runtime admita compute y mesh con subgrupos de 64; con `--mesh-compute` solo necesita
esa capacidad en compute.

Con el controlador de AMD para Windows, `auto` usa wave64 nativo en RDNA1 y RDNA2 si la GPU del runtime lo admite,
aunque otra GPU use subgrupos de 32 o no esté disponible `vulkaninfo`. Así evita un fallo del controlador con
código LDS emulado ([#857](https://github.com/ItsNotRexed/Leonida/issues/857)). En RDNA1 sin mesh nativo comprueba
la capacidad de compute. Puede forzar la emulación con `--wave64-emulate on` o el modo nativo con
`--wave64-emulate off`.

Estas opciones ofrecen alternativas para GPU o controladores con capacidades limitadas. Los detalles y sus
límites están en [`docs/design.md`](../../design.md):

| Opción | Cuándo se usa |
|---|---|
| `--no-float-controls` | Cuando la GPU no admite las capacidades de conservación de valores especiales de coma flotante requeridas por los shaders. |
| `--bary-fallback` | Cuando falta `VK_KHR_fragment_shader_barycentric`. |
| `--texel-offset-fallback` | Cuando falta `VK_KHR_maintenance8` para los desplazamientos de muestreo. |
| `--mesh-compute` | Cuando falta `VK_EXT_mesh_shader`: traduce los shaders mesh como compute. |
| `--f16-interp-fallback` | Interpolación de 16 bits sin entradas `PerVertexKHR`; se activa automáticamente cuando la GPU del runtime no puede usarlas, como ocurre con MoltenVK. |
| `--pc-tail-ubo` | Para GPU cuyo límite de push constants es insuficiente; se activa automáticamente cuando el runtime detecta ese límite. |

El modo wave64 y estas opciones se guardan junto a los shaders. Cambiarlos vuelve a ejecutar la traducción.
`auto` normalmente reutiliza los shaders ya emulados, salvo cuando la alternativa de AMD para Windows requiere
reconstruirlos en modo nativo.

### Reanudar tras una actualización

La compilación registra el comando y la versión de `llvm-mc` que generaron el desensamblado: si cambian, o si la
carpeta de trabajo es anterior a ese registro, vuelve a desensamblar y traducir, conservando las entradas extraídas.
También registra los archivos fuente del traductor; un cambio en ellos repite la traducción a partir del
desensamblado guardado.

Para modelos y texturas registra el commit de `third_party/rpf9-explorer` y el código de `tools/rpf9dump`. Cuando
cambia ese decodificador, elimina la exportación de cada archivo afectado y la genera de nuevo. Las carpetas
antiguas sin esos registros necesitan repetir una vez la traducción o la exportación correspondiente.

### Linux y Wine/Proton

`make_game.py` también funciona en Linux. Las herramientas de diagnóstico Vulkan se compilan de forma nativa
con CMake 3.24+, un compilador C++20 y las cabeceras y el cargador de Vulkan:

```sh
cmake -S runtime -B runtime/build/linux
cmake --build runtime/build/linux --parallel
./runtime/build/linux/vk_pipeline_check /ruta/a/spirv
```

El último comando recibe la carpeta que contiene `index.json` y los shaders traducidos. La etapa `pipelines`
usa `vk_pipeline_check` de `runtime/build/linux` o `runtime/build`, y lo compila antes de iniciar el resto del
trabajo si falta.

El cargador es un programa de Windows y se ejecuta bajo Wine o Proton. En Linux la etapa `runtime` puede
compilarlo de forma cruzada con `clang-cl` y `lld-link` (LLVM 19+, excepto 22.1.7 y 22.1.8) y el CRT de MSVC y el
SDK de Windows preparados por [xwin](https://github.com/Jake-Shadle/xwin), en `XWIN_DIR` o
`~/.local/share/xwin-sdk`. La [guía de pruebas](../../testing.md#the-windows-runtime-on-linux-under-wine-or-proton)
explica la preparación. Si faltan esas herramientas, `runtime` se omite de las etapas predeterminadas; pedirlo
explícitamente enumera lo que falta antes de empezar.

También puede pasar `--runtime-build <carpeta>` con una compilación de Windows (`gta6rt.exe` y
`gta6rt_tls_host.dll`), por ejemplo el artefacto `runtime-windows` del trabajo `windows` del workflow Runtime,
descomprimido. Así evita compilar el runtime; la misma opción funciona en Windows. En Linux el script escribe
`gta6rt.sh`, que lanza `gta6rt.exe` con `wine`, o con el Wine indicado por la variable `WINE` (para Proton, su
`files/bin/wine`).
- **`--wave64-emulate`**: traduce los shaders compute y mesh wave64 de forma exacta cualquiera que sea el tamaño de
  subgrupo de la GPU. Las GPU NVIDIA e Intel usan subgrupos de 32 y lo necesitan. El valor predeterminado, `auto`,
  consulta `vulkaninfo` y activa la emulación salvo que todas las GPU anuncien subgrupos de 64.
- **`--work`**: elija un disco con al menos 400 GB libres. La traducción de shaders es la etapa más larga; por
  defecto usa la mitad de los hilos del procesador (ajustable con `--jobs`).
- **`--wav`**: desactivado por defecto, porque el WAV PCM de 16 bits ocupa varias veces más que el Ogg Opus.
- **Linux y Wine/Proton**: `make_game.py` también funciona en Linux. El cargador es un programa de Windows y se
  ejecuta bajo Wine o Proton. Para la etapa `runtime` puede pasar `--runtime-build <carpeta>` con una compilación
  de Windows (`gta6rt.exe` y `gta6rt_tls_host.dll`): el script escribe entonces `gta6rt.sh`, que lanza
  `gta6rt.exe` con `wine`, o con el Wine indicado por la variable `WINE` (para Proton, su `files/bin/wine`).

## Problemas habituales

- **Usó «Download ZIP» en GitHub**: el ZIP no contiene el submódulo `third_party/rpf9-explorer`. Use
  `git clone --recursive`, o ejecute `git submodule update --init --recursive` en la carpeta del repositorio.
- **`llvm-mc` rechaza `gfx1013` o el destino AMDGPU**: probablemente usa el instalador oficial de LLVM para
  Windows. Ejecute `scripts\build_llvm_mc.ps1`, o apunte `LLVM_MC` al `llvm-mc` de WSL (`llvm-21`).
- **CMake se detiene en «needs the Windows SDK 10.0.26100 or newer»**, o MSVC indica que `Desc_1_2` no es miembro
  de `D3D12_VERSIONED_ROOT_SIGNATURE_DESC`: actualice el SDK de Windows a `10.0.26100` o superior, con el
  Instalador de Visual Studio o con `winget install Microsoft.WindowsSDK.10.0.26100`, y luego borre
  `runtime/build` y repita la configuración.
- **Se quedó sin espacio en `C:`**: pase `--work "D:\leonida-work"` para situar los archivos temporales en otra
  unidad; también puede cambiar `--out` u omitir `assets` con `--stages check,shaders,runtime` (unos 30 GB).
- **Windows se niega a lanzar los programas de prueba recién compilados** («An Application Control policy has
  blocked this file»): es el control inteligente de aplicaciones de Windows 11. Véase
  [`docs/testing.md`](../../testing.md#known-problems).

## Contribuir y ejecutar las pruebas de GPU

1. Revise las pull requests abiertas y la [issue de coordinación][coord] para no rehacer un trabajo en curso.
   Reserve la tarea con `claiming: <ID>`, o con una descripción de una línea si aún no hay identificador.
2. Ejecute las pruebas pertinentes, con entradas sintéticas. Por ejemplo,
   `python -I tests/test_manifest_verifier.py` desde la raíz del repositorio.
3. En la pull request, indique el identificador de la tarea, qué probó, y la GPU y el controlador si ejecutó
   pruebas de GPU. Diga si recibió ayuda de una IA. Las reglas están en
   [`CONTRIBUTING.md`](../../../CONTRIBUTING.md).

La [guía de pruebas de GPU](../../testing.md) explica cómo ejecutarlas sin ningún archivo del juego. Publique sus
resultados en la [issue #166](https://github.com/ItsNotRexed/Leonida/issues/166), un comentario por GPU y por
controlador.

[coord]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22
