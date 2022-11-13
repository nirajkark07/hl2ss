
#include "utilities.h"

//-----------------------------------------------------------------------------
// Logging 
//-----------------------------------------------------------------------------

// OK
void ShowMessage(const char* format, ...)
{
	char* text;
	int len;
	va_list arg_list;
	va_start(arg_list, format);
	len = _vscprintf(format, arg_list) + 2;
	text = (char*)malloc(len);
	if (!text) { return; }
	vsprintf_s(text, len, format, arg_list);
	va_end(arg_list);
	text[len - 2] = '\n';
	text[len - 1] = '\0';
	OutputDebugStringA(text);
	free(text);
}

// OK
void ShowMessage(const wchar_t* format, ...)
{
	wchar_t* text;
	int len;
	va_list arg_list;
	va_start(arg_list, format);
	len = _vscwprintf(format, arg_list) + 2;
	text = (wchar_t*)malloc(len * sizeof(wchar_t));
	if (!text) { return; }
	vswprintf_s(text, len, format, arg_list);
	va_end(arg_list);
	text[len - 2] = L'\n';
	text[len - 1] = L'\0';
	OutputDebugStringW(text);
	free(text);
}

//-----------------------------------------------------------------------------
// Critical Section 
//-----------------------------------------------------------------------------

// OK
CriticalSection::CriticalSection(void* pcs)
{
	m_pcs = pcs;
	if (m_pcs) { EnterCriticalSection(static_cast<CRITICAL_SECTION*>(m_pcs)); }
}

// OK
CriticalSection::~CriticalSection()
{
	if (m_pcs) { LeaveCriticalSection(static_cast<CRITICAL_SECTION*>(m_pcs)); }
}
