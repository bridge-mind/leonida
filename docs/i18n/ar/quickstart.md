<div dir="rtl" lang="ar">

# دليل البدء السريع لمشروع Leonida

الوثائق الإنجليزية هي المرجع المعتمد. يستند هذا الدليل إلى المراجعة `561b65a` ويغطي خطوات البدء الأساسية.
راجع [ملف README بالإنجليزية](../../../README.md#project-status) لمعرفة حالة المشروع الحالية، و[فهرس الوثائق](../../README.md) للمزيد من التفاصيل.

## ما هو Leonida؟

Leonida مشروع لإعادة ترجمة إصدار Xbox Series من Grand Theft Auto VI لتشغيله مباشرةً على Windows.
يُبنى المشروع على جهازك من نسختك الخاصة من اللعبة. تُترجم برامج التظليل من RDNA2 إلى SPIR-V لاستخدامها مع Vulkan،
وتُقرأ الموارد من حزمة اللعبة المحلية، ويُعيد المشروع تنفيذ واجهات النظام التي تحتاجها اللعبة.

لا يحتوي المستودع على اللعبة أو ملفها التنفيذي أو مفاتيحها. تتطلب الأدوات ملفات قابلة للقراءة أصلًا على القرص،
ولا تفك التشفير أو تتجاوز الحماية. احتفظ بملفات اللعبة والملفات الناتجة عن البناء على جهازك. لا ترفقها أو ترفق سجلات تقتبس منها عند الإبلاغ عن مشكلة أو فتح طلب دمج.

## المتطلبات

- نسختك الخاصة من حزمة Xbox Series بالإصدار `1.0.2.1`. اختر المجلد الذي يحتوي على `MicrosoftGame.Config` وأرشيفات `.rpf`.
- Windows 10 أو 11 بمعمارية x64، مع نحو 400 غيغابايت متاحة لملفات العمل. هذا الرقم للبناء الكامل: التشغيل وحده يكفيه `--stages check,shaders,runtime` بنحو 30 غيغابايت، ومرحلة `assets` وحدها تطلب نحو 370 غيغابايت. يمكن وضعها على قرص آخر باستخدام `--work` و`--out`.
- Git و[Python 3.12 أو أحدث](https://www.python.org/downloads/) و[حزمة .NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0).
  عند تثبيت Python على Windows، اختر إضافة `python.exe` إلى `PATH`.
- [حزمة Vulkan SDK](https://vulkan.lunarg.com/) لتوفير `spirv-as` و`spirv-val` و`vulkaninfo`.
- [LLVM 21 أو أحدث](https://github.com/llvm/llvm-project/releases) مع دعم AMDGPU في `llvm-mc`.
  نسخة LLVM الرسمية لـWindows لا تتضمن هذا الدعم. من جذر المستودع، شغّل `scripts/build_llvm_mc.ps1` من PowerShell لبناء `llvm-mc.exe`.
  يتطلب السكربت Git وأدوات C++ في Visual Studio ويضبط `LLVM_MC`. افتح نافذة PowerShell جديدة بعد انتهائه.
  إذا كنت تستخدم [conda](https://docs.conda.io/)، فطريقة أسرع هي `conda install -c conda-forge llvm-tools` (نسخة Windows منه تتضمن هدف AMDGPU؛ استخدم [دليل الاختبارات](../../testing.md) الإصدار 23.1.6)، ثم اضبط `LLVM_MC` على المسار الكامل لملف `llvm-mc.exe` الذي يثبّته (يعرض `where.exe llvm-mc` مساره بعد تنشيط البيئة؛ أما `where` المجرد في PowerShell فهو `Where-Object`).
  إذا ثبّتَّ `llvm-21` على Ubuntu داخل WSL، اضبط `LLVM_MC` في نافذة PowerShell التي ستشغّل منها البناء.

  ```powershell
  $env:LLVM_MC = 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'
  ```

- Visual Studio 2022 مع حزمة **Desktop development with C++** وWindows SDK بالإصدار `10.0.26100` أو أحدث لبناء مكوّن التشغيل.
- [ffmpeg](https://ffmpeg.org/download.html) ضمن `PATH`، وهو اختياري لفك ترميز XMA2 ومطلوب عند استخدام `--wav`.

## البناء على Windows

بعد تثبيت المتطلبات، استنسخ المستودع مع وحدته الفرعية.

```powershell
git clone --recursive https://github.com/ItsNotRexed/Leonida
cd Leonida
```

انقر نقرًا مزدوجًا على `Build.bat`. يبحث البرنامج عن نسختك على الجهاز. إذا وجد أكثر من نسخة فسيطلب منك الاختيار،
وإذا لم يجد أي نسخة فسيطلب مسار المجلد. يتحقق من الأدوات والمساحة المتاحة قبل بدء البناء. تُحفظ النتيجة افتراضيًا
في `Documents\Leonida`، ويمكن استئناف البناء إذا توقف.

### البناء من PowerShell

شغّل الأوامر من جذر المستودع. يعرض `--find` النسخ التي عُثر عليها محليًا، ويُستخدم `--check` لفحص الحزمة فقط.
يضيف `--full-check` مقارنة بصمة SHA-256 لكل ملف، وقد يستغرق ذلك وقتًا. في أمر البناء، استبدل المسارات بالمسارات على جهازك.

```powershell
python -I make_game.py --find
python -I make_game.py --pick
python -I make_game.py --check
python -I make_game.py --check --full-check
python -I make_game.py --game "D:\path\to\package" --out "D:\Games\Leonida" --work "E:\leonida-work"
```

ترتيب المراحل هو `check` ثم `shaders` ثم `assets` ثم `runtime` ثم `pipelines`.
يمكن اختيار مراحل بعينها باستخدام `--stages shaders,assets`. عادةً ما تستغرق ترجمة برامج التظليل أطول وقت،
ويمكن ضبط عدد عملياتها باستخدام `--jobs`. اترك `--wave64-emulate` على القيمة الافتراضية `auto` ما لم تحتج إلى تجاوز اختيار الأداة.
يضيف `--wav` نسخًا بصيغة PCM WAV من مسارات Ogg Opus ويزيد المساحة المطلوبة.

### البناء على Linux

يعمل `make_game.py` على Linux، ويجد `llvm-mc` تلقائيًا إذا كان ضمن `PATH`. تأكد من أن نسخته تدعم AMDGPU.
يعمل محمّل اللعبة، وهو برنامج Windows، عبر Wine أو Proton. إذا كان لديك بناء Windows جاهز للمحمّل،
فمرّر مجلده باستخدام `--runtime-build`. راجع [تعليمات البناء والتشغيل على Linux](../../../README.md#running-the-loader-on-linux).

## حل المشاكل الشائعة

- **الوحدة الفرعية غير موجودة بعد تنزيل ZIP.** استنسخ المستودع باستخدام `--recursive` كما في المثال أعلاه.
  وإذا كان لديك مستودع Git مستنسخ بالفعل، شغّل `git submodule update --init --recursive` من جذره.
- **يرفض `llvm-mc` هدف الترجمة `gfx1013`.** تحقق من استخدام نسخة تدعم AMDGPU، ومن أن `LLVM_MC` يشير إلى الأداة الصحيحة.
- **يطلب CMake حزمة Windows SDK أحدث.** ثبّت الإصدار `10.0.26100` أو أحدث من Visual Studio Installer، ثم أعد تهيئة مجلد البناء.
- **المساحة على قرص النظام غير كافية.** استخدم `--work` لتخزين ملفات العمل على قرص أوسع، و`--out` إذا أردت نقل المخرجات أيضًا.
- **يحظر Windows تشغيل أدوات الاختبار المبنية محليًا.** راجع [المشاكل المعروفة في دليل الاختبارات](../../testing.md#known-problems).

## المساهمة

1. اقرأ [دليل المساهمة](../../../CONTRIBUTING.md) وراجع [مسألة التنسيق][coord] وطلبات الدمج المفتوحة قبل البدء.
2. احجز المهمة بتعليق `claiming: <ID>` في مسألة التنسيق، أو اكتب وصفًا من سطر واحد إن لم يكن لها معرّف بعد.
3. شغّل الاختبارات المناسبة بمدخلات اصطناعية. لتعديلات التوثيق مثلًا، يمكن تشغيل `python -I tests/test_docs_links.py` من جذر المستودع.
4. في طلب الدمج، اذكر معرّف المهمة أو رابط الحجز والاختبارات التي شغّلتها. اذكر البطاقة الرسومية وإصدار تعريفها إذا اختبرت على GPU،
   واذكر أي مساعدة من الذكاء الاصطناعي.

يوضح [دليل اختبارات GPU](../../testing.md) طريقة تشغيل الاختبارات من دون ملفات اللعبة.
يمكن نشر النتائج في [مسألة الاختبارات رقم 166](https://github.com/ItsNotRexed/Leonida/issues/166).

[coord]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22

</div>
