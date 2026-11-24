// Tiny fixture EXE for the recursive-module test. It is never executed; rt::Loader only reads its import table.
#include <windows.h>

extern "C" __declspec(dllimport) int __cdecl tiny_add(int a, int b);
extern "C" __declspec(dllimport) extern int tiny_exported_value;

int main() {
    // Both references keep tiny_dll.dll in the import directory: one function import and one data import.
    return tiny_add(2, 3) == 5 && tiny_exported_value == 1234 ? 0 : 1;
}
