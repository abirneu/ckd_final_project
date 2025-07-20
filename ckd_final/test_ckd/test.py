def patient_profile(request, phone):
    try:
        patient = get_object_or_404(Patient, phone=phone)
        patient.status = 'checked'  # Update status to 'checked'
        patient.save()  # Save the updated status
        
        # Prepare input values from patient data
        input_values = {
            'n1': str(patient.age),
            'n2': str(patient.upper_bp),
            'n3': str(patient.lower_bp),
            'n4': patient.hypertension,
            'n5': patient.diabetes,
            'n6': patient.pedal_edema,
            'n7': patient.anemia,
            'n8': patient.appetite,
        }
        
        # Create a mock request with the input values
        mock_request = type('Request', (), {'GET': input_values, 'method': 'GET'})
        
        # Get the prediction result
        predicted_group = get_prediction_result(mock_request)
        
        context = {
            'patient': patient,
            'predicted_group': predicted_group
        }
        return render(request, 'patient_profile.html', context)
    
    except Exception as e:
        messages.error(request, f"Error loading patient: {str(e)}")
        return redirect('doctor_dashboard')





def get_input_values(request):
    """Process and convert form input values to model-compatible format"""
    vals = []
    for i in range(1, 25):  # For all 24 input fields
        val = request.GET.get(f'n{i}', '').strip()
        
        # Handle specific categorical fields
        if i in [6, 7, 8, 9, 19, 20, 21, 22, 23, 24]:  # Categorical fields
            if val.lower() in ['normal', 'notpresent', 'no', 'good']:
                vals.append(0)
            elif val.lower() in ['abnormal', 'present', 'yes', 'poor']:
                vals.append(1)
            else:
                vals.append(0)  # Default value
        else:
            try:
                vals.append(float(val))
            except ValueError:
                vals.append(0.0)  # Default value for numerical fields
    return vals


def patient_profile(request, phone):
    try:
        patient = get_object_or_404(Patient, phone=phone)
        patient.status = 'checked'
        patient.save()
        
        # Prepare input values from ALL patient data fields
        input_values = {
            'n1': str(patient.age),
            'n2': str(patient.upper_bp),
            'n3': str(patient.lower_bp),
            'n4': patient.hypertension,
            'n5': patient.diabetes,
            'n6': patient.pedal_edema,
            'n7': patient.anemia,
            'n8': patient.appetite,
            'n9': str(patient.specific_gravity),
            'n10': str(patient.albumin),
            'n11': str(patient.sugar),
            'n12': patient.red_blood_cells if hasattr(patient, 'red_blood_cells') else '',
            'n13': patient.pus_cell if hasattr(patient, 'pus_cell') else '',
            'n14': patient.pus_cell_clumps if hasattr(patient, 'pus_cell_clumps') else '',
            'n15': patient.bacteria if hasattr(patient, 'bacteria') else '',
            'n16': str(patient.blood_glucose),
            'n17': str(patient.blood_urea),
            'n18': str(patient.serum_creatinine),
            'n19': str(patient.sodium),
            'n20': str(patient.potassium),
            'n21': str(patient.hemoglobin),
            'n22': str(patient.packed_cell_volume) if hasattr(patient, 'packed_cell_volume') else '0',
            'n23': str(patient.white_blood_cell) if hasattr(patient, 'white_blood_cell') else '0',
            'n24': str(patient.red_blood_cell) if hasattr(patient, 'red_blood_cell') else '0',
            'n25': str(patient.coronary_artery_disease) if hasattr(patient, 'coronary_artery_disease') else '',
        }
        
        # Convert input values to model-compatible format
        vals = get_input_values(input_values)
        
        # Make prediction
        scaled_input = scaler.transform([vals])
        pred = stacking_model.predict(scaled_input)
        result = "CKD" if pred[0] == 1 else "Not CKD"
        
        # Get prediction probabilities
        proba = stacking_model.predict_proba(scaled_input)[0]
        ckd_prob = round(proba[1] * 100, 2)
        not_ckd_prob = round(proba[0] * 100, 2)
        
        context = {
            'patient': patient,
            'prediction_result': result,
            'ckd_probability': ckd_prob,
            'not_ckd_probability': not_ckd_prob,
        }
        return render(request, 'patient_profile.html', context)
    
    except Exception as e:
        messages.error(request, f"Error loading patient: {str(e)}")
        return redirect('doctor_dashboard')

def get_input_values(input_dict):
    """Convert form input values to model-compatible format"""
    vals = []
    for i in range(1, 26):  # For all 25 input fields
        val = input_dict.get(f'n{i}', '').strip().lower()
        
        # Handle specific categorical fields
        if i in [4, 5, 6, 7, 8, 12, 13, 14, 15, 25]:  # Categorical fields
            if val in ['normal', 'notpresent', 'no', 'good']:
                vals.append(0)
            elif val in ['abnormal', 'present', 'yes', 'poor']:
                vals.append(1)
            else:
                vals.append(0)  # Default value
        else:
            try:
                vals.append(float(val))
            except ValueError:
                vals.append(0.0)  # Default value for numerical fields
    return vals


def predict(request):  
    result1 = 'NA'  # Set default result to NA

    if request.method == 'GET' and request.GET:  # Only run prediction if the form is submitted
        # Load the dataset
        dataset = pd.read_csv(r"D:\final_year_project\datasets\Kidney_data.csv")  # Update with your file path

        # Define categorical replacements
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

        # Replace categorical values with numerical values
        for col, mapping in replacements.items():
            if col in dataset.columns:
                dataset[col] = dataset[col].replace(mapping)

        # Convert target column to binary values
        if 'classification' in dataset.columns:
            dataset['classification'] = dataset['classification'].apply(lambda x: 1 if x == 'ckd' else 0)

        # Convert specific object columns to numeric
        for col in ['pcv', 'wc', 'rc']:
            if col in dataset.columns:
                dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

        # Fill missing values with the median
        features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
                    'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad',
                    'appet', 'pe', 'ane']
        for feature in features:
            if feature in dataset.columns:
                dataset[feature].fillna(dataset[feature].median(), inplace=True)

        # Clean column names
        dataset.columns = dataset.columns.str.strip()

        # Define the target column
        target_column = 'classification'

        # Drop unnecessary columns
        columns_to_drop = [col for col in ['Unnamed: 0', 'id'] if col in dataset.columns]
        X = dataset.drop(columns=columns_to_drop + [target_column])
        y = dataset[target_column]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Collect user inputs
        vals = get_input_values(request)

        try:
            scaled_input = scaler.transform([vals])  # Scale input features
            pred = model.predict(scaled_input)
            result1 = "CKD" if pred[0] == 1 else "Not CKD"  # Adjusted for 1 = CKD, 0 = Not CKD
        except Exception as e:
            result1 = f"Error: {str(e)}"  # Display error if something goes wrong

    return render(request, 'predict.html', {"result2": result1})



#temp
def preprocess_ckd_data(dataset):
    """Preprocess the CKD dataset with proper type handling"""
    # Define categorical replacements
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
        'classification': {'ckd\t': 1, 'notckd': 0}
    }

    # Apply categorical replacements
    for col, mapping in replacements.items():
        if col in dataset.columns:
            # dataset[col] = dataset[col].astype(str).str.strip().replace(mapping) =day-15
            dataset[col] = dataset[col].replace(mapping)
    # = day-15
    # Convert target column
    if 'classification' in dataset.columns:
        dataset['classification'] = dataset['classification'].apply(
            lambda x: 1 if x == 'ckd' else 0)

    # Convert numeric columns - handle errors by coercing to NaN
    numeric_cols = ['age', 'lower_bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
                   'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    
    for col in numeric_cols:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

    # Fill missing values only for numeric columns
    # Fill missing values
    features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
               'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 
               'dm', 'cad', 'appet', 'pe', 'ane']
    for feature in features:
        if feature in dataset.columns:
            dataset[feature].fillna(dataset[feature].median(), inplace=True)

    # Clean column names
    dataset.columns = dataset.columns.str.strip()

    return dataset

# def train_and_save_ckd_model():
#     """Train and save the CKD prediction model with proper data handling"""
#     global ckd_model, ckd_scaler
    
#     try:
#         # Load and preprocess data
#         dataset = pd.read_csv(r"test_ckd\datasets\ckr_pred\Kidney_data.csv")
#         dataset = preprocess_ckd_data(dataset)
        
#         # Ensure we have the target column
#         if 'classification' not in dataset.columns:
#             raise ValueError("Target column 'classification' not found in dataset")
        
#         # Define features and target
#         X = dataset.drop(columns=['classification'])
#         y = dataset['classification']

#         # Train-test split
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42)

#         # Scale only numeric features
#         numeric_cols = X_train.select_dtypes(include=['number']).columns
#         scaler = StandardScaler()
#         X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
#         X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

#         # Train model
#         model = LogisticRegression(max_iter=1000, random_state=42)
#         model.fit(X_train[numeric_cols], y_train)
        
#         # Save model
#         model_dir = 'saved_models'
#         os.makedirs(model_dir, exist_ok=True)
#         dump(model, os.path.join(model_dir, 'ckd_model.joblib'))
#         dump(scaler, os.path.join(model_dir, 'ckd_scaler.joblib'))
        
#         return True
#     except Exception as e:
#         print(f"Error in model training: {str(e)}")
#         return False
# = day-15
def train_and_save_model():
    """Train the stacking model and save to disk"""
    global stacking_model, scaler
    
    # Load and preprocess data
    dataset = pd.read_csv(r"test_ckd\datasets\ckr_pred\Kidney_data.csv")
    dataset = preprocess_ckd_data(dataset)
    
    # Define target and features
    target_column = 'classification'
    columns_to_drop = [col for col in ['Unnamed: 0', 'id'] if col in dataset.columns]
    X = dataset.drop(columns=columns_to_drop + [target_column])
    y = dataset[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define base models for stacking
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000))
    ]

    # Define stacking classifier
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1
    )

    # Train stacking model
    stacking_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_acc = accuracy_score(y_train, stacking_model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, stacking_model.predict(X_test_scaled))
    print(f"Training Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Save the trained model and scaler
    model_dir = 'saved_models'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'ckd_stacking_model.joblib')
    scaler_path = os.path.join(model_dir, 'ckd_scaler.joblib')
    
    dump(stacking_model, model_path)
    dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

def get_ckd_prediction(patient_data):
    """Get CKD prediction for patient data with proper type handling"""
    try:
        # Load model and scaler
        model = load(os.path.join('saved_models', 'ckd_model.joblib'))
        scaler = load(os.path.join('saved_models', 'ckd_scaler.joblib'))
        
        # Convert patient data to proper numeric format
        numeric_features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
                          'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
        
        input_values = []
        for feature in numeric_features:
            value = patient_data.get(feature, 0)
            try:
                input_values.append(float(value))
            except (ValueError, TypeError):
                input_values.append(0.0)  # Default value if conversion fails
        
        # Scale and predict
        scaled_input = scaler.transform([input_values])
        pred = model.predict(scaled_input)[0]
        return "CKD" if pred == 1 else "Not CKD"
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Error in prediction"

# Initialize model when module loads
# if not os.path.exists(os.path.join('saved_models', 'ckd_model.joblib')):
#     train_and_save_ckd_model() # type: ignore

# = day-15

# CKD Prediction View
def predict(request):  
    result1 = 'NA'
   

    if request.method == 'GET' and request.GET:
        try:
            # Get user input
            vals = patient_profile(request)
            
            # Make prediction
            scaled_input = scaler.transform([vals])
            pred = stacking_model.predict(scaled_input)
            result1 = "CKD" if pred[0] == 1 else "Not CKD"
            
            
           

        except Exception as e:
            result1 = f"Error: {str(e)}"
            print(f"Prediction error: {e}")

    return render(request, 'predict.html', {
        "result2": result1,
        
    })


# def get_ckd_prediction(request):
#     """Function to get CKD/Not CKD prediction result"""
#     try:
#         # Load dataset
#         dataset = pd.read_csv(r"test_ckd\datasets\ckr_pred\Kidney_data.csv")
        
#         # Preprocess data
#         dataset = preprocess_ckd_data(dataset)

#         # Define features (X) and target (y)
#         X = dataset.drop(columns=['classification'])
#         y = dataset['classification']

#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y
#         )

#         # Scale features
#         numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
#                        'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
#         scaler = StandardScaler()
#         X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
#         X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

#         # Train model
#         model = LogisticRegression(
#             class_weight='balanced',
#             max_iter=1000
#         )
#         model.fit(X_train, y_train)

#         # Get input values from request
#         vals = []
#         for key in [f'n{i}' for i in range(1, 26)]:
#             val = request.GET.get(key, '').strip().lower()
#             if val in ['yes', 'no']:
#                 vals.append(1 if val == 'yes' else 0)
#             elif val in ['good', 'poor']:
#                 vals.append(1 if val == 'good' else 0)
#             elif val in ['normal', 'notpresent']:
#                 vals.append(0)
#             elif val in ['abnormal', 'present']:
#                 vals.append(1)
#             else:
#                 try:
#                     vals.append(float(val))
#                 except ValueError:
#                     vals.append(0.0)
        
#         # Prepare input for prediction
#         input_df = pd.DataFrame([vals], columns=X_train.columns)
#         input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

#         # Predict
#         pred = model.predict(input_df)[0]
#         return "CKD" if pred == 1 else "Not CKD"

#     except Exception as e:
#         print("CKD Prediction Error:", e)
#         return "Error in CKD prediction"

# def preprocess_ckd_data(dataset):
#     """Preprocess the CKD dataset"""
#     replacements = {
#         'rbc': {'normal': 0, 'abnormal': 1},
#         'pc': {'normal': 0, 'abnormal': 1},
#         'pcc': {'notpresent': 0, 'present': 1},
#         'ba': {'notpresent': 0, 'present': 1},
#         'htn': {'no': 0, 'yes': 1},
#         'dm': {'\tyes': 1, ' yes': 1, '\tno': 0, 'no': 0, 'yes': 1},
#         'cad': {'\tno': 0, 'no': 0, 'yes': 1},
#         'appet': {'good': 1, 'poor': 0},
#         'pe': {'no': 0, 'yes': 1},
#         'ane': {'no': 0, 'yes': 1},
#         'classification': {'ckd\t': 1, 'notckd': 0}
#     }

#     for col, mapping in replacements.items():
#         if col in dataset.columns:
#             dataset[col] = dataset[col].replace(mapping)

#     numeric_cols = ['pcv', 'wc', 'rc']
#     for col in numeric_cols:
#         if col in dataset.columns:
#             dataset[col] = pd.to_numeric(dataset[col], errors='coerce')

#     features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
#                'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 
#                'dm', 'cad', 'appet', 'pe', 'ane']
#     for feature in features:
#         if feature in dataset.columns:
#             dataset[feature].fillna(dataset[feature].median(), inplace=True)

#     dataset.columns = dataset.columns.str.strip()
#     return dataset

# def train_and_save_ckd_model():
#     """Train and save the CKD prediction model"""
#     global ckd_model, ckd_scaler
    
#     dataset = pd.read_csv(r"test_ckd\datasets\ckr_pred\Kidney_data.csv")
#     dataset = preprocess_ckd_data(dataset)
    
#     X = dataset.drop(columns=['classification'])
#     y = dataset['classification']

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42)

#     numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 
#                    'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
#     scaler = StandardScaler()
#     X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
#     X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)
    
#     # Save model
#     model_dir = 'saved_models'
#     os.makedirs(model_dir, exist_ok=True)
#     dump(model, os.path.join(model_dir, 'ckd_model.joblib'))
#     dump(scaler, os.path.join(model_dir, 'ckd_scaler.joblib'))

# def load_ckd_model():
#     """Load the pre-trained CKD model"""
#     global ckd_model, ckd_scaler
    
#     model_dir = 'saved_models'
#     model_path = os.path.join(model_dir, 'ckd_model.joblib')
#     scaler_path = os.path.join(model_dir, 'ckd_scaler.joblib')
    
#     if os.path.exists(model_path) and os.path.exists(scaler_path):
#         ckd_model = load(model_path)
#         ckd_scaler = load(scaler_path)
#         return True
#     return False

# # Initialize model when module loads
# if not load_ckd_model():
#     train_and_save_ckd_model()


#new



#end ckd predict section

#grp predict section start

