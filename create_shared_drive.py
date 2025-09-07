from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import os
import io

def create_shared_drive():
    # Define the required scopes
    SCOPES = ['https://www.googleapis.com/auth/drive']
    
    # Get path to service account file from environment variable
    SERVICE_ACCOUNT_FILE = 'google_service_account.json'
    
    # Load credentials from the service account file
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    
    # Create Drive API service
    service = build('drive', 'v3', credentials=credentials)
    
    try:
        # Create a new shared drive
        drive_metadata = {
            'name': 'PDF Bot Storage'
        }
        
        shared_drive = service.drives().create(
            body=drive_metadata,
            requestId='pdf-bot-drive'  # Unique identifier for the request
        ).execute()
        
        print(f"Created shared drive: {shared_drive.get('name')} (ID: {shared_drive.get('id')})")
        
        # Try to create a file in the shared drive
        file_metadata = {
            'name': 'test.txt',
            'parents': [shared_drive.get('id')]
        }
        
        # Create file content
        content = 'This is a test file'
        fh = io.BytesIO(content.encode('utf-8'))
        media = MediaIoBaseUpload(fh, mimetype='text/plain', resumable=True)
        
        # Create the file
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            supportsAllDrives=True,
            fields='id, name'
        ).execute()
        
        print(f"Created test file: {file.get('name')} (ID: {file.get('id')})")
        
        # List files in the shared drive
        results = service.files().list(
            driveId=shared_drive.get('id'),
            corpora='drive',
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields='files(id, name)'
        ).execute()
        
        print("\nFiles in shared drive:")
        for item in results.get('files', []):
            print(f"{item['name']} ({item['id']})")
            
        return shared_drive.get('id')
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == '__main__':
    create_shared_drive()
