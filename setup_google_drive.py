from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import os
import io
import pickle

def setup_google_drive():
    """
    Sets up Google Drive access with OAuth2 authentication.
    This will open a web browser for you to authenticate with your Google account.
    """
    
    # If modifying these scopes, delete the token.pickle file.
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds = None

    # The token.pickle file stores the user's access and refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                print("⚠️ credentials.json file not found!")
                print("Please follow these steps:")
                print("1. Go to https://console.cloud.google.com")
                print("2. Select your project")
                print("3. Go to APIs & Services > Credentials")
                print("4. Click 'Create Credentials' > 'OAuth client ID'")
                print("5. Choose 'Desktop app' as application type")
                print("6. Download the JSON file and save it as 'credentials.json' in this directory")
                return
                
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    try:
        # Create Google Drive service
        service = build('drive', 'v3', credentials=creds)
        
        # Create a test folder
        folder_metadata = {
            'name': 'PDF Bot Storage',
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        folder = service.files().create(
            body=folder_metadata,
            fields='id'
        ).execute()
        
        folder_id = folder.get('id')
        print(f"✅ Created folder 'PDF Bot Storage' (ID: {folder_id})")
        
        # Create a test file in the folder
        file_metadata = {
            'name': 'test.txt',
            'parents': [folder_id]
        }
        
        content = 'This is a test file'
        fh = io.BytesIO(content.encode('utf-8'))
        media = MediaIoBaseUpload(fh, mimetype='text/plain', resumable=True)
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        print(f"✅ Created test file (ID: {file.get('id')})")
        
        # Update folder permissions to make it accessible
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        
        service.permissions().create(
            fileId=folder_id,
            body=permission
        ).execute()
        
        print(f"✅ Folder permissions updated")
        print(f"\nYou can now use this folder ID in your application: {folder_id}")
        print("\nTo use this in your application:")
        print("1. Update your .env file with:")
        print(f"GOOGLE_DRIVE_FOLDER_ID={folder_id}")
        
        return folder_id
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == '__main__':
    setup_google_drive()
