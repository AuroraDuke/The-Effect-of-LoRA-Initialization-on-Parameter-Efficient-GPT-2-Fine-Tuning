# Creates a zip file for submission on Gradescope.

import os
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

required_files = [p for p in os.listdir(PROJECT_ROOT) if p.endswith('.py')] + \
                 [os.path.join('predictions', p) for p in os.listdir(os.path.join(PROJECT_ROOT, 'predictions'))]

for root, _, files in os.walk(os.path.join(PROJECT_ROOT, 'src')):
    for filename in files:
        if filename.endswith('.py'):
            path = os.path.join(root, filename)
            required_files.append(os.path.relpath(path, PROJECT_ROOT))

def main():
    aid = 'cs224n_default_final_project_submission'

    with zipfile.ZipFile(os.path.join(PROJECT_ROOT, f"{aid}.zip"), 'w') as zz:
        for file in required_files:
            zz.write(os.path.join(PROJECT_ROOT, file), os.path.join(".", file))
    print(f"Submission zip file created: {aid}.zip")

if __name__ == '__main__':
    main()
