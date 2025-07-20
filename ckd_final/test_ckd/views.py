# At module level (top of views.py)
grp_model = None
ckd_model = None
grp_scaler = None
ckd_scaler = None

def load_models():
    global grp_model, ckd_model, grp_scaler, ckd_scaler
    # Load group prediction model and scaler
    # Load CKD prediction model and scaler
from pyexpat.errors import messages
from venv import logger
from django.shortcuts import redirect, render
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import RobustScaler  # CHANGED: StandardScaler → RobustScaler

from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os
from django.contrib import messages

from test_ckd.forms import PatientForm
from test_ckd.models import Patient

from django.shortcuts import  get_object_or_404

from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseServerError, QueryDict
from .models import Patient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from django.contrib import messages
import re
import re
from django.contrib import messages
from django.shortcuts import render, redirect
from .models import Patient  # Adjust if needed


from .models import  Profile
# Create your views here.
def home(request):
    return render(request, 'home.html')

#new
# views.py
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login

from django.contrib import messages

from django.contrib.auth.models import User

from django.shortcuts import render, redirect

from .models import  Profile
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_POST
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import render, redirect

from django.contrib.auth.decorators import login_required, user_passes_test

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        reg = request.POST['reg']
        email = request.POST['email']
        phone = request.POST['phone']
        dob = request.POST['date_of_birth']
        password = request.POST['password']
        address = request.POST['address']
        user_type = request.POST['diabetes']  # This comes from your select field

        if User.objects.filter(username=username).exists():
            return render(request, 'register.html', {'error': 'Username already taken'})
        
        user = User.objects.create_user(username=username, email=email, password=password)
        Profile.objects.create(
            user=user, 
            reg=reg, 
            phone=phone, 
            date_of_birth=dob,
            address=address,
            user_type=user_type
        )

        # Authenticate and login the user after registration
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            if user_type == 'doctor':
                return redirect('doctor_dashboard')
            else:
                return redirect('patient_info')

    return render(request, 'register.html')

from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from .models import Profile

from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect
from .models import Profile

def login_view(request):
    if request.method == 'POST':
        phone = request.POST['phone']
        password = request.POST['password']

        try:
            profile = Profile.objects.get(phone=phone)
            user = authenticate(request, username=profile.user.username, password=password)
            
            if user:
                login(request, user)
                if profile.user_type == 'doctor':
                    return redirect('doctor_dashboard')
                else:
                    return redirect('patient_info')
            else:
                return render(request, 'login.html', {'error': 'Invalid credentials'})

        except Profile.DoesNotExist:
            return render(request, 'login.html', {'error': 'User not found'})

    return render(request, 'login.html')

def doctor_required(view_func):
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        try:
            profile = request.user.profile
            if profile.user_type != 'doctor':
                return redirect('patient_info')
            return view_func(request, *args, **kwargs)
        except Profile.DoesNotExist:
            return redirect('login')
    return wrapper

def staff_required(view_func):
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        try:
            profile = request.user.profile
            if profile.user_type != 'staff':
                return redirect('doctor_dashboard')
            return view_func(request, *args, **kwargs)
        except Profile.DoesNotExist:
            return redirect('login')
    return wrapper

@doctor_required
def doctor_dashboard(request):
    # Your existing doctor dashboard code
    pass

@staff_required
def patient_info_id(request):
    # Your existing patient info code
    pass

def logout_view(request):
    logout(request)
    return redirect('home')

# Other Views (Example for Dashboard)
def doctor_dashboard(request):
    return render(request, 'doctor_dashboard.html')


def patient_info(request):
    return render(request, 'patient_info.html')


def home(request):
    return render(request, 'home.html')


# def login(request):
#     return render(request, 'signin.html')
#     pass
# def signup(request):

#     return render(request,'signup.html' )

def predict(request):
    # Logic for prediction goes here
    # For now, we will just render a simple template
    return render(request, 'predict.html')

def group(request):
    return render(request, 'grp.html')  # Make sure this template exists as 'grp.html'
def doctor_dashboard(request):
    patients = Patient.objects.all()
    checked_count = patients.filter(status='checked').count()
    pending_count = patients.filter(status='pending').count()
    
    context = {
        'patients': patients,
        'checked_count': checked_count,
        'pending_count': pending_count,
    }
    return render(request, 'doctor_dashboard.html', context)



#patient register dont touch this part
def patient_info_id(request):
    
    if request.method == 'POST':
        form_data = {
            field: request.POST.get(field, '').strip()
            for field in [
                'name', 'phone', 'age', 'upper_bp', 'lower_bp',
                'hypertension', 'diabetes', 'coronary_artery_disease',
                'specific_gravity', 'albumin', 'sugar', 'red_blood_cells',
                'pus_cell', 'pus_cell_clumps', 'bacteria',
                'blood_glucose', 'blood_urea', 'serum_creatinine',
                'sodium', 'potassium', 'hemoglobin', 'packed_cell_volume',
                'white_blood_cell', 'red_blood_cell',
                'pedal_edema', 'anemia', 'appetite'
            ]
        }

        # Required field checks
        errors = []
        required_fields = [
            'name', 'phone', 'age'
        ]

        for field in required_fields:
            if not form_data[field]:
                errors.append(f"{field.replace('_', ' ').title()} is required")

        if form_data['age'] and (not form_data['age'].isdigit() or not (1 <= int(form_data['age']) <= 120)):
            errors.append("Age must be between 1-120")

        phone_regex = r'^01[3-9]\d{8}$'
        if form_data['phone'] and not re.match(phone_regex, form_data['phone']):
            errors.append("Phone number must be a valid Bangladeshi number (e.g., 01712345678)")

        if errors:
            for error in errors:
                messages.error(request, error)
            return render(request, 'patient_info.html', {'form_data': form_data})

        # Convert numeric fields safely
        numeric_fields = [
            'age', 'upper_bp', 'lower_bp', 'specific_gravity', 'albumin', 'sugar',
            'blood_glucose', 'blood_urea', 'serum_creatinine', 'sodium', 'potassium',
            'hemoglobin', 'packed_cell_volume', 'white_blood_cell', 'red_blood_cell'
        ]

        for field in numeric_fields:
            try:
                form_data[field] = float(form_data[field]) if form_data[field] != '' else None
            except ValueError:
                form_data[field] = None

        # Optional string fields
        optional_fields = [
            'coronary_artery_disease', 'red_blood_cells', 'pus_cell',
            'pus_cell_clumps', 'bacteria'
        ]

        for field in optional_fields:
            if form_data[field] == '':
                form_data[field] = None

        try:
            # Create and save patient
            patient = Patient.objects.create(
                name=form_data['name'],
                phone=form_data['phone'],
                age=int(form_data['age']),
                upper_bp=int(form_data['upper_bp']),
                lower_bp=int(form_data['lower_bp']),
                hypertension=form_data['hypertension'],
                diabetes=form_data['diabetes'],
                coronary_artery_disease=form_data['coronary_artery_disease'],
                specific_gravity=form_data['specific_gravity'],
                albumin=form_data['albumin'],
                sugar=form_data['sugar'],
                red_blood_cells=form_data['red_blood_cells'],
                pus_cell=form_data['pus_cell'],
                pus_cell_clumps=form_data['pus_cell_clumps'],
                bacteria=form_data['bacteria'],
                blood_glucose=form_data['blood_glucose'],
                blood_urea=form_data['blood_urea'],
                serum_creatinine=form_data['serum_creatinine'],
                sodium=form_data['sodium'],
                potassium=form_data['potassium'],
                hemoglobin=form_data['hemoglobin'],
                packed_cell_volume=form_data['packed_cell_volume'],
                white_blood_cell=form_data['white_blood_cell'],
                red_blood_cell=form_data['red_blood_cell'],
                pedal_edema=form_data['pedal_edema'],
                anemia=form_data['anemia'],
                appetite=form_data['appetite'],
             
            )

            messages.success(request, 'Patient data saved successfully!')
            return redirect('patient_info')

        except Exception as e:
            messages.error(request, f'Error saving patient data: {str(e)}')
            return render(request, 'patient_info.html', {'form_data': form_data})

    return render(request, 'patient_info.html')



#ckd predict section
# Global variables for model and scaler
stacking_model = None
scaler = None





# Global variables for model and scaler
# ckd_model = None
# ckd_scaler = None


def patient_profile(request, phone):
    try:
        # Fetch patient information
        patient = get_object_or_404(Patient, phone=phone)
        patient.status = 'checked'
        patient.save()
        
        
        # Prepare input values for CKD prediction (specific to CKD-related features)
        input_values_ckd = {
            'age': str(patient.age),
            'lower_bp': str(patient.lower_bp),
            'sg': str(patient.specific_gravity),
            'al': str(patient.albumin),
            'su': str(patient.sugar),
            'rbc': patient.red_blood_cells if hasattr(patient, 'red_blood_cells') else '',
            'pc': patient.pus_cell if hasattr(patient, 'pus_cell') else '',
            'pcc': patient.pus_cell_clumps if hasattr(patient, 'pus_cell_clumps') else '',
            'ba': patient.bacteria if hasattr(patient, 'bacteria') else '',
            'bgr': str(patient.blood_glucose),
            'bu': str(patient.blood_urea),
            'sc': str(patient.serum_creatinine),
            'sod': str(patient.sodium),
            'pot': str(patient.potassium),
            'hemo': str(patient.hemoglobin),
            'pcv': str(patient.packed_cell_volume) if hasattr(patient, 'packed_cell_volume') else '0',
            'wc': str(patient.white_blood_cell) if hasattr(patient, 'white_blood_cell') else '0',
            'rc': str(patient.red_blood_cell) if hasattr(patient, 'red_blood_cell') else '0',
            'htn': patient.hypertension,
            'dm': patient.diabetes,
            'cad': str(patient.coronary_artery_disease) if hasattr(patient, 'coronary_artery_disease') else '',
            'appet': patient.appetite,
            'pe': patient.pedal_edema,
            'ane': patient.anemia,
        }

        # Prepare input values for group prediction (specific to group-related features)
        input_values_grp = {
            'n1': str(patient.age),
            'n2': str(patient.upper_bp),
            'n3': str(patient.lower_bp),
            'n4': patient.hypertension,
            'n5': patient.diabetes,
            'n6': patient.pedal_edema,
            'n7': patient.anemia,
            'n8': patient.appetite,
        }

        # Get the CKD prediction result
        ckd_result = predict_ckd(input_values_ckd)  # CKD Prediction
        
        # Get the group prediction result
        predicted_group = get_prediction_result(input_values_grp)  # Group Prediction
        
        # Pass both results to the context
        context = {
            'patient': patient,
            'result2': ckd_result,  # CKD prediction result
            'predicted_group': predicted_group  # Group prediction result
        }

        return render(request, 'patient_profile.html', context)
    
    except Exception as e:
        messages.error(request, f"Error loading patient: {str(e)}")
        return redirect('doctor_dashboard')






from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def get_prediction_result(input_values):
    """Function to get prediction result"""
    try:
        # Load dataset
        dataset_grp = pd.read_excel(r"test_ckd\datasets\grp_pred\ckd_grp_predct_dataset.xlsx")

        # Clean & replace 'Classification'
        dataset_grp['Classification'] = dataset_grp['Classification'].str.strip().replace({
            'Group 1': 1, 'Group 2': 2, 'Group 3': 3, 'Group 4': 4,
            'Panel A': 5, 'Panel B': 6, 'Panel C': 7, 'Panel D': 8
        })

        # Drop rows with invalid 'Classification'
        dataset_grp = dataset_grp.dropna(subset=['Classification'])
        dataset_grp['Classification'] = dataset_grp['Classification'].astype(int)

        # Handle categorical features
        categorical_cols = {
            'HTN': {'no': 0, 'yes': 1},
            'DM': {'no': 0, 'yes': 1},
            'pe': {'no': 0, 'yes': 1},
            'ane': {'no': 0, 'yes': 1},
            'appt': {'good': 1, 'poor': 0}
        }

        for col, mapping in categorical_cols.items():
            if col in dataset_grp.columns:
                dataset_grp[col] = dataset_grp[col].str.strip().str.lower().replace(mapping).astype(int)

        # Ensure numeric features
        numeric_cols = ['Age', 'Upper BP', 'Lower BP']
        for col in numeric_cols:
            dataset_grp[col] = pd.to_numeric(dataset_grp[col], errors='coerce').fillna(0)

        # Define features (X) and target (y)
        X = dataset_grp.drop(columns=['Id', 'Classification'])
        y = dataset_grp['Classification']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        # Train model
        model = LogisticRegression(
            multi_class='ovr',
            class_weight='balanced',
            max_iter=1000
        )
        model.fit(X_train, y_train)

        # Get input values from the passed dictionary (input_values)
        vals = []
        for key in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']:
            val = input_values.get(key, '').strip().lower()
            if val in ['yes', 'no']:
                vals.append(1 if val == 'yes' else 0)
            elif val in ['good', 'poor']:
                vals.append(1 if val == 'good' else 0)
            else:
                try:
                    vals.append(float(val))
                except ValueError:
                    vals.append(0.0)

        # Prepare input for prediction
        input_df = pd.DataFrame([vals], columns=X_train.columns)
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Predict
        pred = model.predict(input_df)[0]

        # Map prediction to group/panel
        group_map = {
            1: 'Group 1', 2: 'Group 2', 3: 'Group 3', 4: 'Group 4',
            5: 'Panel A', 6: 'Panel B', 7: 'Panel C', 8: 'Panel D'
        }
        return group_map.get(pred, 'Unknown')

    except Exception as e:
        print("Prediction Error:", e)
        return "Error in prediction"




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ============ DATA PREPROCESSING ============

# Load your dataset
dataset = pd.read_csv(r"test_ckd/datasets/ckr_pred/Kidney_data.csv")

replacements = {
    'rbc': {'normal': 0, 'abnormal': 1},
    'pc': {'normal': 0, 'abnormal': 1},
    'pcc': {'notpresent': 0, 'present': 1},
    'ba': {'notpresent': 0, 'present': 1},
    'htn': {'no': 0, 'yes': 1},
    'dm': {'\tyes': 1, ' yes': 1, '\tno': 0, 'no': 0, 'yes': 1},
    'cad': {'\tno': 0, 'no': 0, 'yes': 1},
    'appet': {'good': 1, 'poor': 0},
    'pe': {'no': 0, 'yes': 1},
    'ane': {'no': 0, 'yes': 1},
    'classification': {'ckd\t': 'ckd'}
}

for col, mapping in replacements.items():
    if col in dataset.columns:
        dataset[col] = dataset[col].replace(mapping)
if 'classification' in dataset.columns:
    dataset['classification'] = dataset['classification'].apply(lambda x: 1 if x == 'ckd' else 0)
for col in ['pcv', 'wc', 'rc']:
    if col in dataset.columns:
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
dataset.columns = dataset.columns.str.strip()

target_column = 'classification'
columns_to_drop = [col for col in ['Unnamed: 0', 'id'] if col in dataset.columns]
X = dataset.drop(columns=columns_to_drop + [target_column])
y = dataset[target_column]

# ============ TRAIN-TEST SPLIT ============
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============ IMPUTE MISSING VALUES ============
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# ============ FEATURE SCALING ============
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# ============ ENSEMBLE MODEL ============
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# ============ SAVE MODELS ============
os.makedirs('saved_models', exist_ok=True)
joblib.dump(model, 'saved_models/best_ckd_model.joblib')
joblib.dump(imputer, 'saved_models/ckd_imputer.joblib')
joblib.dump(scaler, 'saved_models/ckd_scaler.joblib')
print("All models saved in 'saved_models' folder.")

import joblib
import numpy as np

def predict_ckd(input_values):
    try:
        # Load pre-trained models
        model = joblib.load('saved_models/best_ckd_model.joblib')
        imputer = joblib.load('saved_models/ckd_imputer.joblib')
        scaler = joblib.load('saved_models/ckd_scaler.joblib')

        # Define your feature columns in the same order as training
        feature_columns = [
            'age', 'lower_bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
            'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
            'appet', 'pe', 'ane'
        ]

        # Same categorical replacements as before
        replacements = {
            'rbc': {'normal': 0, 'abnormal': 1},
            'pc': {'normal': 0, 'abnormal': 1},
            'pcc': {'notpresent': 0, 'present': 1},
            'ba': {'notpresent': 0, 'present': 1},
            'htn': {'no': 0, 'yes': 1},
            'dm': {'\tyes': 1, ' yes': 1, '\tno': 0, 'no': 0, 'yes': 1},
            'cad': {'\tno': 0, 'no': 0, 'yes': 1},
            'appet': {'good': 1, 'poor': 0},
            'pe': {'no': 0, 'yes': 1},
            'ane': {'no': 0, 'yes': 1}
        }

        # Prepare input in correct order and format
        input_data = []
        for col in feature_columns:
            if col in replacements:
                val = input_values.get(col, '')
                input_data.append(replacements[col].get(val, 0))
            else:
                try:
                    input_data.append(float(input_values.get(col, 0)))
                except:
                    input_data.append(0)
        input_array = np.array(input_data).reshape(1, -1)

        # Impute missing values, then scale, then predict
        imputed_input = imputer.transform(input_array)
        scaled_input = scaler.transform(imputed_input)
        pred = model.predict(scaled_input)
        return "CKD" if pred[0] == 1 else "Not CKD"

    except Exception as e:
        return f"Error in prediction: {str(e)}"

