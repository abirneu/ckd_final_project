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


def get_prediction_result(request):
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

        # Get input values directly from request.GET
        vals = []
        for key in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']:
            val = request.GET.get(key, '').strip().lower()
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



   