"""
Final Submission Packager and Quality Assurance Script.
Assembles the exact required challenge zip structure:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   └── [all source files]
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import shutil
import zipfile
import subprocess

def package_submission(team_name: str = "EntityResolvers"):
    print("=== FINAL SUBMISSION PACKAGING & QA ===")
    
    root_dir = "/Users/madanthambisetty/Downloads/student_resource"
    output_dir = os.path.join(root_dir, "output")
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
    doc_file = os.path.join(root_dir, "Documentation_template.md")
    code_dir = os.path.join(root_dir, "code", "business_entity_resolution")
    
    # Check that required files exist
    print("Checking required files...")
    assert os.path.isfile(matching_file), f"Missing {matching_file}"
    assert os.path.isfile(candidate_file), f"Missing {candidate_file}"
    assert os.path.isfile(doc_file), f"Missing {doc_file}"
    assert os.path.isdir(code_dir), f"Missing {code_dir}"
    assert os.path.isfile(os.path.join(code_dir, "README.md")), "Missing code README.md"
    assert os.path.isfile(os.path.join(code_dir, "requirements.txt")), "Missing requirements.txt"
    assert os.path.isdir(os.path.join(code_dir, "src")), "Missing code/src"
    
    # 1. Run official validator
    print("\nRunning Official Challenge Validator...")
    cmd = [
        "python3", "utils/validate_submission.py",
        "--matching", "output/matching_results.tsv",
        "--candidate", "output/candidate_pairs.tsv",
        "--test-dir", "dataset/test"
    ]
    res = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    if res.returncode != 0:
        print("ERROR: Official validator failed!")
        return False
        
    # 2. Build clean temporary folder structure
    staging_dir = os.path.join(root_dir, f"{team_name}_submission_staging")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)
    
    # Create subdirectories
    staging_output = os.path.join(staging_dir, "output")
    staging_code = os.path.join(staging_dir, "code", "business_entity_resolution")
    staging_src = os.path.join(staging_code, "src")
    os.makedirs(staging_output)
    os.makedirs(staging_src)
    
    # Copy files
    print("\nCopying files to staging directory...")
    shutil.copy2(matching_file, staging_output)
    shutil.copy2(candidate_file, staging_output)
    shutil.copy2(doc_file, staging_dir)
    shutil.copy2(os.path.join(code_dir, "README.md"), staging_code)
    shutil.copy2(os.path.join(code_dir, "requirements.txt"), staging_code)
    
    for fname in os.listdir(os.path.join(code_dir, "src")):
        src_path = os.path.join(code_dir, "src", fname)
        if os.path.isfile(src_path) and not fname.startswith('.'):
            shutil.copy2(src_path, staging_src)
            
    # 3. Create zip archive
    zip_filename = f"{team_name}_submission.zip"
    zip_path = os.path.join(root_dir, zip_filename)
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print(f"\nCreating submission archive: {zip_filename}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(staging_dir):
            for file in files:
                if file.startswith('.') or file.endswith('.pyc'):
                    continue
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, staging_dir)
                zipf.write(file_path, rel_path)
                
    # Clean staging directory
    shutil.rmtree(staging_dir)
    
    # 4. Verify ZIP structure
    print(f"\nVerifying contents of {zip_filename}:")
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        for info in zipf.infolist():
            print(f"  {info.filename} ({info.file_size:,} bytes)")
            
    print(f"\nSUCCESS: {zip_filename} created and verified successfully!")
    return True

if __name__ == "__main__":
    package_submission()
