; ============================================================================
; Klippinge Investment Trading Terminal - Inno Setup Installer Script
; ============================================================================
;
; Requires: Inno Setup 6+ (https://jrsoftware.org/isinfo.php)
;
; Build:
;   1. Install Inno Setup 6
;   2. Open this file in Inno Setup Compiler
;   3. Click Build → Compile
;   OR from command line: ISCC build/installer.iss

#define MyAppName "Klippinge Investment Trading Terminal"
#define MyAppShortName "KlippingeTrading"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Klippinge Investment"
#define MyAppURL "https://github.com/YOUR_USERNAME/klippinge-trading"
#define MyAppExeName "KlippingeTrading.exe"
#define MyAppIcon "..\logo.ico"

[Setup]
; Unique AppId - generate a new GUID at https://www.guidgenerator.com/
AppId={{cbb90478-0555-4960-9849-2721615041a5}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={autopf}\{#MyAppShortName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Output settings
OutputDir=..\dist
OutputBaseFilename=KlippingeTrading-v{#MyAppVersion}-Setup
SetupIconFile={#MyAppIcon}
; Compression
Compression=lzma2/ultra64
SolidCompression=yes
; Windows version requirement
MinVersion=10.0
; Privileges
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; UI
WizardStyle=modern
WizardSizePercent=110
; Uninstaller
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "swedish"; MessagesFile: "compiler:Languages\Swedish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startupicon"; Description: "Start with Windows"; GroupDescription: "Other options:"; Flags: unchecked

[Files]
; Main application directory (from PyInstaller dist output)
Source: "..\dist\KlippingeTrading\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; Optional: Add to startup
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
    ValueType: string; ValueName: "{#MyAppShortName}"; \
    ValueData: """{app}\{#MyAppExeName}"""; Flags: uninsdeletevalue; Tasks: startupicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; \
    Flags: nowait postinstall skipifsilent

[Code]
// Close the app before installing an update
function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
begin
  Result := True;
  // Try to close running instance gracefully
  if CheckForMutexes('{#MyAppShortName}_Mutex') then
  begin
    if MsgBox('{#MyAppName} is currently running. ' +
              'It will be closed to continue installation.',
              mbConfirmation, MB_OKCANCEL) = IDOK then
    begin
      Exec('taskkill', '/f /im {#MyAppExeName}', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
      Sleep(1000);
    end
    else
      Result := False;
  end;
end;
