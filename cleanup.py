import os

# Files to keep (essential files)
essential_files = {
    'test_app.py',           # Main working app
    'perfect_model.py',      # ML model
    'perfect_model.pkl',     # Trained model
    'Suicide_Detection.csv', # Dataset
    'requirements.txt',      # Dependencies
    'README.md',            # Documentation
    'Procfile',             # Deployment
    'render.yaml',          # Deployment config
    '.gitignore',           # Git config
    '.gitattributes',       # Git config
    'templates',            # HTML folder
    'cleanup.py'            # This script
}

# Files to delete (duplicates/unused)
files_to_delete = []

for file in os.listdir('.'):
    if os.path.isfile(file) and file not in essential_files:
        files_to_delete.append(file)

print("Files that can be safely deleted:")
for i, file in enumerate(files_to_delete, 1):
    print(f"{i}. {file}")

if files_to_delete:
    print(f"\nTotal: {len(files_to_delete)} duplicate/unused files found")
    choice = input("\nDelete all these files? (y/n): ").lower()
    
    if choice == 'y':
        deleted_count = 0
        for file in files_to_delete:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {file}: {e}")
        
        print(f"\nCleanup complete! Deleted {deleted_count} files")
    else:
        print("Cleanup cancelled")
else:
    print("No duplicate files found. Project is already clean!")

print("\nEssential files kept:")
for file in sorted(essential_files):
    if os.path.exists(file):
        print(f"  {file}")