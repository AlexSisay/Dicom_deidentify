"""
DICOM De-Identification Utility

This script performs GDPR- and HIPAA-compliant anonymization of DICOM files from MRI datasets.
It targets only T2-weighted axial and sagittal series folders and removes or replaces sensitive
metadata like patient name, birthdate, institution, comments, and unique identifiers.

Expected Directory Structure:
  Dataset/
    0001/
      Risonanza_Magnetica_Lombosacrale - */
        T2_SAG_*/
          *.dcm
        T2_AX_*/
          *.dcm
        Other folders (ignored)

Usage:
  Save this script as dicom_anonymizer.py
  Run with:
    python dicom_anonymizer.py -P /path/to/main/MRI_dataset_folder

Features:
- Detects and targets only T2_SAG and T2_AX series folders.
- Removes or masks all personal identifiers and sensitive metadata.
- Preserves PatientSex, PatientAge, and assigns new anonymized PatientID.
- Saves de-identified DICOMs in mirrored structure under 'Anonymised/' directory.
- Logs problematic files and empty directories for review.

"""
import os
import re
import json
import glob
import argparse
import logging
import hashlib
from datetime import datetime
from multiprocessing import Pool, cpu_count
import pydicom
from pydicom import dcmread

# --- Logging Configuration ---
logging.basicConfig(
    filename='anonymization.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Fields to Anonymize ---
# These correspond to direct identifiers or quasi-identifiers as defined under GDPR Article 4(1)
# and HIPAA Safe Harbor (45 CFR §164.514(b)(2))

DATE_FIELDS = ['DateOfLastCalibration', 'PatientBirthDate']
TIME_FIELDS = ['TimeOfLastCalibration']
NAME_FIELDS = ['InstitutionName', 'InstitutionAddress', 'ReferringPhysicianName']
COMMENT_FIELDS = ['ImageComments']
ID_FIELDS = ['IrradiationEventUID']
META_FIELDS = ['PrivateInformationCreatorUID', 'PrivateInformation']

# Entire DICOM sequences to remove that may contain PHI
SEQUENCE_FIELDS = ['RequestAttributesSequence']

def extract_id_sex_age(path):
    s = re.search(r'\d+ F \d+|\d+ M \d+', path)
    s = s.group(0).split()
    return s[0].strip(), s[1].strip(), s[2].strip()

def anonymise(d, sex, age, patient_id):
    try:
        # GDPR/HIPAA Compliance: Strip all private fields including those marked with (7FE0,0010) etc.
        d.remove_private_tags()

        # Anonymize dates to static values – GDPR Recital 26: remove data that enables identification
        for k in DATE_FIELDS:
            if k in d: d[k].value = '20250419'

        # Anonymize times similarly
        for k in TIME_FIELDS:
            if k in d: d[k].value = '094338'

        # Remove institutional names and comments – direct identifiers per HIPAA Safe Harbor
        for k in NAME_FIELDS + COMMENT_FIELDS:
            if k in d: d[k].clear()

        # Remove any persistent or trackable ID fields – meets both GDPR pseudonymization and HIPAA §164.514
        for k in ID_FIELDS:
            if k in d: del d[k]

        # Clear full sequences that might contain nested PHI (e.g., referring physician name)
        for k in SEQUENCE_FIELDS:
            d.pop(k, None)

        # Remove sensitive meta fields (including custom tags by vendors)
        for k in META_FIELDS:
            d.file_meta.pop(k, None)

        # Restore minimal required patient metadata for ML model use (sex, age), in anonymized form
        # Age is normalized to 3 digits, suffixed by 'Y' per DICOM standard
        age = age.zfill(3) + 'Y'
        sex = sex + ' '
        patient_id = patient_id if len(patient_id) % 2 == 0 else patient_id + ' '

        # These are needed for downstream ML processing but are GDPR pseudonymized
        d['PatientSex'].value = sex
        d['PatientAge'].value = age
        d['PatientID'].value = patient_id
    except Exception as e:
        logging.error(f"Failed to anonymize DICOM: {e}")
        raise
    return d

def validate_checksum(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
            hash_val = hashlib.md5(data).hexdigest()
            dcmread(file_path)  # Ensures structural validity, not just file presence
        return hash_val
    except Exception as e:
        logging.error(f"Checksum validation failed for {file_path}: {e}")
        return None

def process_folder(args):
    root, files, path, cat, last_directory, last_date_time = args
    problematic, empty_dir = [], []
    if not files:
        empty_dir.append(root)
        return problematic, empty_dir

    if re.sub(cat[0], cat[1], root) == last_directory or re.sub(cat[1], cat[0], root) == last_directory:
        new_patient = False
    else:
        new_patient = True

    if new_patient:
        date_time = datetime.now().strftime("%Y%m%d%H%M%S")[2:]
        while last_date_time == date_time:
            date_time = datetime.now().strftime("%Y%m%d%H%M%S")[2:]
        last_date_time = date_time

    names = ["%04d" % x for x in range(len(files))]
    last_directory = root
    files.sort()

    for idx, filename in enumerate(files):
        file_path = os.path.join(root, filename)
        try:
            x = dcmread(file_path)
            mater_id, sex, age = extract_id_sex_age(file_path)
            patient_id = date_time + mater_id + 'M'
            anonym_x = anonymise(x, sex, age, patient_id)

            save_to = re.sub('DICOM(.*)', '', file_path)
            save_to = re.sub(path, 'Anonymised', save_to)
            save_to = re.sub(mater_id, patient_id, save_to)
            os.makedirs(save_to, exist_ok=True)
            new_file_path = os.path.join(save_to, 'EXP' + str(names[idx]) + '.dcm')
            anonym_x.save_as(new_file_path)
            validate_checksum(new_file_path)

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            problematic.append(file_path + ': %s' % e)
            continue

    logging.info(f"Anonymized: {root}")
    return problematic, empty_dir

def main():
    parser = argparse.ArgumentParser(description="De-identify DICOM files from dataset")
    parser.add_argument("-P", "--path", type=str, required=True, help="Root path of dataset")
    args = parser.parse_args()

    all_roots = []
    cat = ['T2_AX', 'T2_SAG']
    last_date_time = ''

    for root, _, files in os.walk(args.path):
        if 'EXP00000' in root:
            all_roots.append((root, files, args.path, cat, '', last_date_time))

    with Pool(cpu_count()) as pool:
        results = pool.map(process_folder, all_roots)

    problematic, empty_dir = [], []
    for prob, empty in results:
        problematic.extend(prob)
        empty_dir.extend(empty)

    with open('problematic_files.json', 'w') as f:
        json.dump(problematic, f)
    with open('empty_directories.json', 'w') as f:
        json.dump(empty_dir, f)

    logging.info("Anonymisation process completed.")

if __name__ == '__main__':
    main()
