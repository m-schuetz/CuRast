// 
// Drag and Dropping adapted form windows classic samples:
//     https://github.com/microsoft/Windows-classic-samples/blob/main/LICENSE
//     https://github.com/microsoft/Windows-classic-samples/tree/7af17c73750469ed2b5732a49e5cb26cbb716094/Samples/Win7Samples/winui/shell/appplatform/DragDropVisuals
// 
// LICENSE: MIT (https://github.com/microsoft/Windows-classic-samples/blob/7af17c73750469ed2b5732a49e5cb26cbb716094/LICENSE)
//
//

#ifdef _WIN32

#define GLFW_EXPOSE_NATIVE_WIN32
#define NOMINMAX

#include <vector>
#include <string>
#include <sstream>
#include <functional>
#include <print>
#include "locale"
#include <codecvt>
#include <cstddef>
#include <wchar.h>
#include <stddef.h>
#include <cstdint>

#include "GLFW\glfw3.h"
#include "GLFW\glfw3native.h"

using std::vector;
using std::string;
using std::function;
using std::println;


	#include "windows.h"
	#include "Commctrl.h"
	#include <strsafe.h>
	#include <commoncontrols.h>
	#include <windowsx.h>           // for WM_COMMAND handling macros
	#include <shlobj.h>             // shell stuff
	#include <shlwapi.h>            // QISearch, easy way to implement QI
	#include <propkey.h>
	#include <propvarutil.h>
	#include <objbase.h>

	vector<function<void(string)>> dragEnterCallbacks;
	vector<function<void(string)>> dragDropCallbacks;
	vector<function<void(int32_t, int32_t)>> dragOverCallbacks;

	// helper to convert a data object with HIDA format or folder into a shell item
	// note: if the data object contains more than one item this function will fail
	// if you want to operate on the full selection use SHCreateShellItemArrayFromDataObject

	HRESULT CreateItemFromObject(IUnknown* punk, REFIID riid, void** ppv)
	{
		*ppv = NULL;

		PIDLIST_ABSOLUTE pidl;
		HRESULT hr = SHGetIDListFromObject(punk, &pidl);
		
		if (SUCCEEDED(hr)){
			hr = SHCreateItemFromIDList(pidl, riid, ppv);
			ILFree(pidl);
		}

		return hr;
	}


	class CDragDropHelper : public IDropTarget{
	
	public:
		CDragDropHelper() : 
			_hrOleInit(OleInitialize(0)), 
			_hwndRegistered(NULL), 
			_dropImageType(DROPIMAGETYPE::DROPIMAGE_NOIMAGE), 
			_pszDropTipTemplate(NULL)
		{
			
		}

		~CDragDropHelper(){

		}

		// IDropTarget
		IFACEMETHODIMP DragEnter(IDataObject *pdtobj, DWORD /* grfKeyState */, POINTL pt, DWORD *pdwEffect)
		{

			IShellItem *psi;
			HRESULT hr = CreateItemFromObject(pdtobj, IID_PPV_ARGS(&psi));
			
			if (SUCCEEDED(hr)){
				PWSTR pszName;
				hr = psi->GetDisplayName(SIGDN_DESKTOPABSOLUTEPARSING, &pszName);

				if (SUCCEEDED(hr)){
					std::wstringstream ss;
					ss << pszName;

					using convert_type = std::codecvt_utf8<wchar_t>;
					std::wstring_convert<convert_type, wchar_t> converter;

					std::string path = converter.to_bytes(ss.str());

					for(auto& callback : dragEnterCallbacks){
						callback(path);
					}

						
					CoTaskMemFree(pszName);
				}
				psi->Release();
			}

			return S_OK;
		}

		IFACEMETHODIMP DragOver(DWORD grfKeyState, POINTL pt, DWORD *pdwEffect){

			for(auto& callback : dragOverCallbacks){
				int32_t x = pt.x;
				int32_t y = pt.y;
				callback(x, y);
			}

			return S_OK;
		}

		IFACEMETHODIMP DragLeave(){
			
			// TODO: remove model froms cene

			return S_OK;
		}

		IFACEMETHODIMP Drop(IDataObject *pdtobj, DWORD grfKeyState, POINTL pt, DWORD *pdwEffect){

			IShellItem *psi;
			HRESULT hr = CreateItemFromObject(pdtobj, IID_PPV_ARGS(&psi));
			
			if (SUCCEEDED(hr)){
				PWSTR pszName;
				hr = psi->GetDisplayName(SIGDN_DESKTOPABSOLUTEPARSING, &pszName);

				if (SUCCEEDED(hr)){
					std::wstringstream ss;
					ss << pszName;

					using convert_type = std::codecvt_utf8<wchar_t>;
					std::wstring_convert<convert_type, wchar_t> converter;

					std::string path = converter.to_bytes(ss.str());

					for(auto& callback : dragDropCallbacks){
						callback(path);
					}
						
					CoTaskMemFree(pszName);
				}
				psi->Release();
			}

			return S_OK;
		}
		IFACEMETHODIMP QueryInterface(REFIID riid, void **ppv){
			return S_OK;
		}

		IFACEMETHODIMP_(ULONG) AddRef(){
			return InterlockedIncrement(&_cRef);
		}

		IFACEMETHODIMP_(ULONG) Release(){
			long cRef = InterlockedDecrement(&_cRef);

			return cRef;
		}

	private:
		DROPIMAGETYPE _dropImageType;
		PCWSTR _pszDropTipTemplate;
		HWND _hwndRegistered;
		HRESULT _hrOleInit;

		long _cRef;
		HWND _hdlg;
		IShellItemArray *_psiaDrop;
	};

	CDragDropHelper helper;

	void init(HWND handle){
		static bool initialized = false;

		if(!initialized){
			RegisterDragDrop(handle, &helper);

			initialized = true;
		}
	}

	void installDragEnterHandler(GLFWwindow* window, function<void(string)> callback){

		HWND handle = glfwGetWin32Window(window);

		init(handle);

		dragEnterCallbacks.push_back(callback);
	}

	void installDragDropHandler(GLFWwindow* window, function<void(string)> callback){

		HWND handle = glfwGetWin32Window(window);

		init(handle);

		dragDropCallbacks.push_back(callback);
	}

	void installDragOverHandler(GLFWwindow* window, function<void(int32_t, int32_t)> callback){

		HWND handle = glfwGetWin32Window(window);

		init(handle);

		dragOverCallbacks.push_back(callback);
	}
#endif