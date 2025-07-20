from django.db import models
from django.core.validators import RegexValidator, MinValueValidator, MaxValueValidator
from django.contrib.auth.models import User
class Patient(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('checked', 'Checked'),
    ]
    
    # Personal Information
    phone = models.CharField(
        primary_key=True,
        max_length=11,
        validators=[
            RegexValidator(
                regex=r'^01[3-9]\d{8}$',
                message="Phone number must be a valid Bangladeshi mobile number (e.g. 01712345678)"
            )
        ],
        unique=True,
        error_messages={
            'unique': "A patient with this phone number already exists."
        }
    )
    name = models.CharField(max_length=100)
    age = models.PositiveIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(120)]
    )
    
    # Vital Signs
    upper_bp = models.PositiveIntegerField(verbose_name="Upper BP (mmHg)")
    lower_bp = models.PositiveIntegerField(verbose_name="Lower BP (mmHg)")
    hypertension = models.CharField(max_length=3, choices=[('yes', 'Yes'), ('no', 'No')])
    diabetes = models.CharField(max_length=3, choices=[('yes', 'Yes'), ('no', 'No')])
    coronary_artery_disease = models.CharField(max_length=3, choices=[('yes', 'Yes'), ('no', 'No')], blank=True , null=True)
    
    # Urine Analysis
    specific_gravity = models.FloatField(
        validators=[MinValueValidator(1.0), MaxValueValidator(1.03)],
        null=True, blank=True
    )
    albumin = models.PositiveIntegerField(null=True, blank=True)
    sugar = models.PositiveIntegerField(null=True, blank=True)
    red_blood_cells = models.CharField(
        max_length=10, 
        choices=[('normal', 'Normal'), ('abnormal', 'Abnormal')],
        blank=True, null=True
    )
    pus_cell = models.CharField(
        max_length=10,
        choices=[('normal', 'Normal'), ('abnormal', 'Abnormal')],
        blank=True, null=True
    )
    pus_cell_clumps = models.CharField(
        max_length=10,
        choices=[('present', 'Present'), ('notpresent', 'Not Present')],
        blank=True, null=True
    )
    bacteria = models.CharField(
        max_length=10,
        choices=[('present', 'Present'), ('notpresent', 'Not Present')],
        blank=True, null=True
    )
    
    # Blood Tests
    blood_glucose = models.FloatField(verbose_name="Blood Glucose (mg/dL)", null=True, blank=True)
    blood_urea = models.FloatField(verbose_name="Blood Urea (mg/dL)", null=True, blank=True)
    serum_creatinine = models.FloatField(verbose_name="Serum Creatinine (mg/dL)", null=True, blank=True)
    sodium = models.FloatField(verbose_name="Sodium (mEq/L)", null=True, blank=True)
    potassium = models.FloatField(verbose_name="Potassium (mEq/L)", null=True, blank=True)
    hemoglobin = models.FloatField(verbose_name="Hemoglobin (g/dL)", null=True, blank=True)
    packed_cell_volume = models.FloatField(null=True, blank=True)
    white_blood_cell = models.FloatField(verbose_name="WBC (cells/cmm)", null=True, blank=True)
    red_blood_cell = models.FloatField(verbose_name="RBC (millions/cmm)", null=True, blank=True)
    
    # Clinical Findings
    pedal_edema = models.CharField(max_length=3, choices=[('yes', 'Yes'), ('no', 'No')], null=True, blank=True )
    anemia = models.CharField(max_length=3, choices=[('yes', 'Yes'), ('no', 'No')])
    appetite = models.CharField(max_length=10, choices=[('good', 'Good'), ('poor', 'Poor')], null=True, blank=True)
    
    # System
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} ({self.phone})"

    class Meta:
        verbose_name = "Patient"
        verbose_name_plural = "Patients"
        ordering = ['-created_at']



#new
from django.conf import settings
from django.db import models

class Profile(models.Model):
    USER_TYPE_CHOICES = (
        ('doctor', 'Doctor'),
        ('staff', 'Staff'),
    )
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    phone = models.CharField(max_length=15, unique=True)
    date_of_birth = models.DateField(null=True, blank=True)
    reg = models.CharField(max_length=20, null=True, blank=True)
    address = models.TextField(null=True, blank=True)
    user_type = models.CharField(
    max_length=10,
    choices=(('doctor', 'Doctor'), ('staff', 'Staff')),
    null=True,
    blank=True,
)



    def __str__(self):
        return self.user.username

    def __str__(self):
        return self.user.username

    @property
    def is_doctor(self):
        return self.user_type == 'doctor'

    @property
    def is_staff(self):
        return self.user_type == 'staff'