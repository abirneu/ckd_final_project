from django.apps import AppConfig


class TestCkdConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'test_ckd'

#new
# accounts/apps.py
from django.apps import AppConfig

class TestCkdConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'test_ckd'  # Python path to application
    