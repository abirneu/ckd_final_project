from django.contrib import admin
from .models import Patient

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('name', 'phone', 'age', 'status', 'created_at', 'hypertension', 'diabetes')
    list_filter = ('status', 'hypertension', 'diabetes', 'coronary_artery_disease', 'anemia', 'created_at')
    search_fields = ('name', 'phone', 'age')
    list_per_page = 20
    ordering = ('-created_at',)
    date_hierarchy = 'created_at'
    
    # Group fields in a more logical way
    fieldsets = (
        ('Personal Information', {
            'fields': ('name', 'phone', 'age', 'status')
        }),
        ('Vital Signs', {
            'fields': (
                ('upper_bp', 'lower_bp'),
                ('hypertension', 'diabetes', 'coronary_artery_disease'),
            )
        }),
        ('Clinical Findings', {
            'fields': ('pedal_edema', 'anemia', 'appetite')
        }),
        ('Urine Analysis', {
            'fields': (
                ('specific_gravity', 'albumin', 'sugar'),
                ('red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria'),
            ),
            'classes': ('collapse',)  # Makes this section collapsible
        }),
        ('Blood Tests', {
            'fields': (
                ('blood_glucose', 'blood_urea', 'serum_creatinine'),
                ('sodium', 'potassium', 'hemoglobin'),
                ('packed_cell_volume', 'white_blood_cell', 'red_blood_cell'),
            ),
            'classes': ('collapse',)  # Makes this section collapsible
        }),
    )
    
    # Add readonly fields for timestamps
    readonly_fields = ('created_at', 'updated_at')
    
    # Customize how fields are displayed in the list view
    def get_list_display(self, request):
        if request.user.is_superuser:
            return ('name', 'phone', 'age', 'status', 'created_at', 'hypertension', 'diabetes')
        return ('name', 'phone', 'age', 'status', 'created_at')
    
    # Add action to change status
    actions = ['mark_as_checked']
    
    def mark_as_checked(self, request, queryset):
        queryset.update(status='checked')
    mark_as_checked.short_description = "Mark selected patients as checked"



#new
# admin.py
from .models import Profile
# admin.site.register(Task)
admin.site.register(Profile)