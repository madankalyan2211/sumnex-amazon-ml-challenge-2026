import os
import shutil
import subprocess

def deliver():
    print("=== FINALIZING & DELIVERING SUBMISSION FILES ===")
    root_dir = "/Users/madanthambisetty/Downloads/student_resource"
    downloads_dir = "/Users/madanthambisetty/Downloads"
    
    matching_src = os.path.join(root_dir, "output", "matching_results.tsv")
    candidate_src = os.path.join(root_dir, "output", "candidate_pairs.tsv")
    zip_src = os.path.join(root_dir, "EntityResolvers_submission.zip")
    
    # 1. Package submission zip
    print("\n[Step 1] Running package_submission.py...")
    cmd = ["python3", "package_submission.py"]
    res = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("ERROR packaging submission:", res.stderr)
        return False
        
    # 2. Copy to Downloads
    print("\n[Step 2] Copying files to Downloads directory...")
    if os.path.exists(matching_src):
        shutil.copy2(matching_src, os.path.join(downloads_dir, "matching_results.tsv"))
        print(f"  ✓ Copied matching_results.tsv to {downloads_dir}/matching_results.tsv")
    if os.path.exists(candidate_src):
        shutil.copy2(candidate_src, os.path.join(downloads_dir, "candidate_pairs.tsv"))
        print(f"  ✓ Copied candidate_pairs.tsv to {downloads_dir}/candidate_pairs.tsv")
    if os.path.exists(zip_src):
        shutil.copy2(zip_src, os.path.join(downloads_dir, "EntityResolvers_submission.zip"))
        print(f"  ✓ Copied EntityResolvers_submission.zip to {downloads_dir}/EntityResolvers_submission.zip")
        
    print("\nSUCCESS: All files delivered to Downloads!")
    return True

if __name__ == "__main__":
    deliver()
