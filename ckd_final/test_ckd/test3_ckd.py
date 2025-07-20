def patient_profile(request, phone):
    try:
        patient = get_object_or_404(Patient, phone=phone)
        patient.status = 'checked'
        patient.save()
        
        # Prepare input values from patient data
        input_values = {
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
        
        # Get the CKD prediction result
        ckd_result = predict_ckd(input_values)  # Call the prediction function directly
        
        context = {
            'patient': patient,
            'result2': ckd_result  # Changed from 'ckd_prediction' to 'result2' to match your template
        }
        print("result is: ", ckd_result)
        return render(request, 'patient_profile.html', context)
        
    except Exception as e:
        messages.error(request, f"Error loading patient: {str(e)}")
        return redirect('doctor_dashboard')

def predict_ckd(input_values):
    try:
        # Load the dataset
        dataset = pd.read_csv(r"test_ckd\datasets\ckr_pred\Kidney_data.csv")

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
        features = ['age', 'lower_bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu',
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

        # Prepare input data in the correct order
        input_data = []
        for col in X.columns:
            # Convert categorical fields to numerical
            if col in replacements:
                if col in ['htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
                    # These are boolean fields from your Patient model
                    input_data.append(1 if input_values.get(col, '') == 'yes' else 0)
                else:
                    # Other categorical fields
                    val = input_values.get(col, '')
                    if val in replacements[col]:
                        input_data.append(replacements[col][val])
                    else:
                        input_data.append(0)  # default value
            else:
                # Numerical fields
                try:
                    input_data.append(float(input_values.get(col, 0)))
                except:
                    input_data.append(0)  # default value if conversion fails

        # Make prediction
        scaled_input = scaler.transform([input_data])
        pred = model.predict(scaled_input)
        return "CKD" if pred[0] == 1 else "Not CKD"
        
    except Exception as e:
        return f"Error in prediction: {str(e)}"