### **ПРОМПТ: Python Ассистент (REFERENCE v2)**
Этот файл содержит расширенные примеры и детальную документацию для `python-assistant-core.md`
---
## РАСШИРЕННЫЕ ПРИМЕРЫ CODE_SAFETY
### **Pydantic валидация (детально)**
**Базовое использование:**
```python
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
class UserRegistration(BaseModel):
    username: str = Field(min_length=3, max_length=20, pattern="^[a-zA-Z0-9_]+$")
    email: EmailStr
    age: int = Field(ge=13, le=120)
    password: str = Field(min_length=8)
    phone: Optional[str] = None
    @validator('password')
    def validate_password_strength(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        return v
    class Config:
        # Запретить дополнительные поля
        extra = 'forbid'
# Использование
def register_user(data: dict) -> User:
    try:
        validated_data = UserRegistration(**data)
        logger.info(f"User data validated: {validated_data.username}")
        # Создание пользователя с валидированными данными
        return User.create(
            username=validated_data.username,
            email=validated_data.email,
            age=validated_data.age
        )
    except ValidationError as e:
        logger.error(f"Validation failed: {e.json()}")
        raise ValueError(f"Invalid user data: {e}")
```
**Вложенные модели:**
```python
from pydantic import BaseModel
from typing import List
class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str = Field(pattern=r"^\d{5}(-\d{4})?$")
class OrderItem(BaseModel):
    product_id: int = Field(gt=0)
    quantity: int = Field(gt=0, le=1000)
    price: float = Field(gt=0)
class Order(BaseModel):
    user_id: int
    items: List[OrderItem] = Field(min_items=1, max_items=100)
    shipping_address: Address
    billing_address: Optional[Address] = None
    @validator('billing_address', always=True)
    def set_billing_address(cls, v, values):
        # Если billing_address не указан, использовать shipping_address
        return v or values.get('shipping_address')
```
**Альтернатива без Pydantic:**
```python
import re
from typing import Dict, Any
def validate_user_registration(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ручная валидация если Pydantic недоступен."""
    errors = []
    # Username
    username = data.get('username', '')
    if not username or len(username) < 3 or len(username) > 20:
        errors.append("Username must be 3-20 characters")
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        errors.append("Username can only contain letters, numbers, underscore")
    # Email
    email = data.get('email', '')
    if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email):
        errors.append("Invalid email format")
    # Age
    age = data.get('age')
    if not isinstance(age, int) or age < 13 or age > 120:
        errors.append("Age must be between 13 and 120")
    # Password
    password = data.get('password', '')
    if len(password) < 8:
        errors.append("Password must be at least 8 characters")
    if not any(c.isupper() for c in password):
        errors.append("Password must contain uppercase letter")
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain digit")
    if errors:
        logger.error(f"Validation errors: {errors}")
        raise ValueError("; ".join(errors))
    return {
        'username': username,
        'email': email,
        'age': age,
        'password': password
    }
```
---
### **Mutable Default Arguments (детально)**
**Почему это опасно:**
```python
# ❌ ОПАСНО - все вызовы используют ОДИН список!
def add_to_list(value, my_list=[]):
    my_list.append(value)
    return my_list
result1 = add_to_list(1)  # [1]
result2 = add_to_list(2)  # [1, 2] ← СЮРПРИЗ!
result3 = add_to_list(3)  # [1, 2, 3] ← Неожиданное поведение
print(result1 is result2)  # True - это один объект!
```
**Правильные паттерны:**
```python
from typing import Optional, List, Dict
# ✅ Паттерн 1: None + создание внутри
def add_to_list(value, my_list: Optional[List] = None) -> List:
    if my_list is None:
        my_list = []
    my_list.append(value)
    return my_list
# ✅ Паттерн 2: Sentinel value
_SENTINEL = object()
def process_data(data, cache=_SENTINEL) -> dict:
    if cache is _SENTINEL:
        cache = {}
    # ...
    return cache
# ✅ Паттерн 3: Для словарей
def configure(options: Optional[Dict] = None) -> Dict:
    if options is None:
        options = {}
    # Установка defaults
    options.setdefault('timeout', 30)
    options.setdefault('retries', 3)
    return options
# ✅ Паттерн 4: Immutable defaults разрешены
def greet(name: str, prefix: str = "Hello") -> str:
    return f"{prefix}, {name}!"
def calculate(x: int, multiplier: int = 2) -> int:
    return x * multiplier
```
**Другие mutable типы:**
```python
# ❌ Все эти - ОПАСНО
def bad_dict(key, value, storage={}): ...
def bad_set(item, items=set()): ...
def bad_custom(obj, container=MyClass()): ...
# ✅ Правильно
def good_dict(key, value, storage: Optional[Dict] = None): 
    if storage is None:
        storage = {}
    ...
def good_set(item, items: Optional[Set] = None):
    if items is None:
        items = set()
    ...
def good_custom(obj, container: Optional[MyClass] = None):
    if container is None:
        container = MyClass()
    ...
```
---
### **SQL-инъекции (детально)**
**Опасные паттерны:**
```python
# ❌ КРИТИЧЕСКИ ОПАСНО
user_input = "1 OR 1=1; DROP TABLE users; --"
# String concatenation
query = "SELECT * FROM users WHERE id = " + user_input
# Результат: DELETE всех данных!
# f-strings
query = f"SELECT * FROM users WHERE name = '{name}'"
# Атака: name = "'; DROP TABLE users; --"
# % formatting
query = "SELECT * FROM users WHERE id = %s" % user_input
# Тоже опасно!
# .format()
query = "SELECT * FROM users WHERE id = {}".format(user_input)
# Опасно!
```
**Безопасные паттерны:**
**1. SQLite / PostgreSQL (psycopg2):**
```python
import sqlite3
import logging
logger = logging.getLogger(__name__)
def get_user_by_id(user_id: int) -> Optional[dict]:
    """Безопасный запрос с параметризацией."""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    try:
        # ✅ Параметризованный запрос
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        if result:
            logger.debug(f"User found: {user_id}")
            return {
                'id': result[0],
                'name': result[1],
                'email': result[2]
            }
        return None
    except Exception as e:
        logger.exception(f"Database error: {e}")
        raise
    finally:
        conn.close()
def search_users(name_pattern: str) -> List[dict]:
    """Поиск с LIKE - тоже параметризованный."""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    try:
        # ✅ Параметризация работает и с LIKE
        cursor.execute(
            "SELECT * FROM users WHERE name LIKE ?",
            (f"%{name_pattern}%",)
        )
        results = cursor.fetchall()
        logger.info(f"Found {len(results)} users matching '{name_pattern}'")
        return results
    finally:
        conn.close()
```
**2. SQLAlchemy ORM:**
```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
Base = declarative_base()
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
engine = create_engine('sqlite:///database.db')
Session = sessionmaker(bind=engine)
def get_user_by_id(user_id: int) -> Optional[User]:
    """ORM автоматически параметризует."""
    session = Session()
    try:
        # ✅ Безопасно - SQLAlchemy параметризует
        user = session.query(User).filter(User.id == user_id).first()
        logger.debug(f"Query executed for user_id: {user_id}")
        return user
    except Exception as e:
        logger.exception(f"Database error: {e}")
        raise
    finally:
        session.close()
def search_users_by_name(name: str) -> List[User]:
    """Поиск через ORM."""
    session = Session()
    try:
        # ✅ Безопасно
        users = session.query(User).filter(User.name.like(f"%{name}%")).all()
        logger.info(f"Found {len(users)} users")
        return users
    finally:
        session.close()
```
**3. Динамические запросы (осторожно!):**
```python
from typing import Dict, Any
def build_search_query(filters: Dict[str, Any]) -> tuple:
    """Построение динамического запроса безопасно."""
    # Whitelist разрешенных полей
    ALLOWED_FIELDS = {'name', 'email', 'age', 'city'}
    conditions = []
    params = []
    for field, value in filters.items():
        # Валидация поля
        if field not in ALLOWED_FIELDS:
            logger.warning(f"Attempted to filter by invalid field: {field}")
            continue
        conditions.append(f"{field} = ?")
        params.append(value)
    if not conditions:
        query = "SELECT * FROM users"
    else:
        query = "SELECT * FROM users WHERE " + " AND ".join(conditions)
    logger.debug(f"Built query: {query} with {len(params)} params")
    return query, tuple(params)
# Использование
filters = {'name': 'John', 'city': 'New York'}
query, params = build_search_query(filters)
cursor.execute(query, params)  # ✅ Безопасно
```
---
### **Секреты и конфигурация (детально)**
**Структура проекта с .env:**
```
project/
├── src/
│   ├── main.py
│   └── config.py
├── .env                    ← Локальные секреты (в .gitignore!)
├── .env.example            ← Пример для других разработчиков
├── .gitignore              ← Обязательно!
├── requirements.txt
└── README.md
```
**Файл `.env`:**
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
DATABASE_PASSWORD=super_secret_password
# API Keys
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
STRIPE_API_KEY=sk_live_xxxxxxxxxxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxxxxxxxxxx
# AWS
AWS_ACCESS_KEY_ID=AKIAxxxxxxxxxxxxx
AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxx
AWS_REGION=us-east-1
# Application
SECRET_KEY=django-insecure-xxxxxxxxxxxxx
DEBUG=False
ALLOWED_HOSTS=example.com,www.example.com
# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=app_specific_password
```
**Файл `.env.example` (для репозитория):**
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
DATABASE_PASSWORD=your_database_password_here
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
STRIPE_API_KEY=your_stripe_api_key_here
STRIPE_WEBHOOK_SECRET=your_webhook_secret_here
# AWS
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1
# Application
SECRET_KEY=generate_random_secret_key
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1
# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@example.com
SMTP_PASSWORD=your_email_password_here
```
**Файл `.gitignore`:**
```
# Environment variables
.env
.env.local
.env.*.local
# Python
__pycache__/
*.py[cod]
*.so
.Python
venv/
env/
# IDE
.vscode/
.idea/
*.swp
# Logs
*.log
main.log
# OS
.DS_Store
Thumbs.db
```
**Конфигурация `config.py`:**
```python
import os
from typing import Optional
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)
# Загрузка .env файла
load_dotenv()
class Config:
    """Централизованная конфигурация приложения."""
    # Database
    DATABASE_URL: str = os.getenv('DATABASE_URL', '')
    DATABASE_PASSWORD: str = os.getenv('DATABASE_PASSWORD', '')
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    STRIPE_API_KEY: Optional[str] = os.getenv('STRIPE_API_KEY')
    STRIPE_WEBHOOK_SECRET: Optional[str] = os.getenv('STRIPE_WEBHOOK_SECRET')
    # AWS
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION: str = os.getenv('AWS_REGION', 'us-east-1')
    # Application
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'INSECURE_DEFAULT_KEY')
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    ALLOWED_HOSTS: list = os.getenv('ALLOWED_HOSTS', 'localhost').split(',')
    # Email
    SMTP_HOST: str = os.getenv('SMTP_HOST', 'localhost')
    SMTP_PORT: int = int(os.getenv('SMTP_PORT', '587'))
    SMTP_USER: Optional[str] = os.getenv('SMTP_USER')
    SMTP_PASSWORD: Optional[str] = os.getenv('SMTP_PASSWORD')
    @classmethod
    def validate(cls) -> None:
        """Валидация обязательных переменных."""
        required = {
            'DATABASE_URL': cls.DATABASE_URL,
            'SECRET_KEY': cls.SECRET_KEY,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            error_msg = f"Missing required environment variables: {', '.join(missing)}"
            logger.critical(error_msg)
            raise ValueError(error_msg)
        if cls.SECRET_KEY == 'INSECURE_DEFAULT_KEY':
            logger.warning("Using default SECRET_KEY - this is insecure!")
        logger.info("Configuration validated successfully")
    @classmethod
    def log_config(cls) -> None:
        """Логирование конфигурации (без секретов!)."""
        logger.info("="*50)
        logger.info("Application Configuration")
        logger.info("="*50)
        logger.info(f"DEBUG: {cls.DEBUG}")
        logger.info(f"ALLOWED_HOSTS: {cls.ALLOWED_HOSTS}")
        logger.info(f"DATABASE_URL: {cls.DATABASE_URL[:20]}... (truncated)")
        logger.info(f"AWS_REGION: {cls.AWS_REGION}")
        logger.info(f"SMTP_HOST: {cls.SMTP_HOST}")
        logger.info(f"OPENAI_API_KEY: {'Set' if cls.OPENAI_API_KEY else 'Not set'}")
        logger.info(f"STRIPE_API_KEY: {'Set' if cls.STRIPE_API_KEY else 'Not set'}")
        logger.info("="*50)
# Использование
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    try:
        Config.validate()
        Config.log_config()
    except ValueError as e:
        logger.critical(f"Configuration error: {e}")
        exit(1)
```
**Использование в приложении:**
```python
from config import Config
import logging
logger = logging.getLogger(__name__)
def send_email(to: str, subject: str, body: str) -> None:
    """Отправка email используя конфигурацию."""
    import smtplib
    from email.mime.text import MIMEText
    if not Config.SMTP_USER or not Config.SMTP_PASSWORD:
        logger.error("SMTP credentials not configured")
        raise ValueError("Email not configured")
    logger.debug(f"Sending email to {to}")
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = Config.SMTP_USER
    msg['To'] = to
    try:
        with smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT) as server:
            server.starttls()
            server.login(Config.SMTP_USER, Config.SMTP_PASSWORD)
            server.send_message(msg)
        logger.info(f"Email sent successfully to {to}")
    except Exception as e:
        logger.exception(f"Failed to send email: {e}")
        raise
```
---
## РАСШИРЕННЫЕ ПРИМЕРЫ NAMING
### Специальные случаи именования
```python
# Приватные атрибуты
class User:
    def __init__(self):
        self._internal_cache = {}  # Protected
        self.__private_attr = None  # Name mangling
    def _internal_method(self):  # Protected метод
        pass
# Type Variables
from typing import TypeVar, Generic
T = TypeVar("T")
UserT = TypeVar("UserT", bound="User")
class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
# Instance и class methods
class MyClass:
    def instance_method(self):  # первый параметр: self
        pass
    @classmethod
    def class_method(cls):  # первый параметр: cls
        pass
    @staticmethod
    def static_method():  # нет первого параметра
        pass
```
---
## РАСШИРЕННЫЕ ПРИМЕРЫ FILE_NAMING_IN_SUBDIRS
### Детальная структура проекта
```
my_project/
├── src/
│   ├── main.py                    # Корневой уровень - без префикса
│   ├── config.py
│   └── app.py
│
├── models/                        # Все файлы с префиксом "model"
│   ├── model_base.py
│   ├── model_user.py
│   ├── model_product.py
│   └── model_order.py
│
├── services/                      # Все файлы с префиксом "service"
│   ├── service_auth.py
│   ├── service_payment.py
│   └── service_notification.py
│
├── repositories/                  # Все файлы с префиксом "repository"
│   ├── repository_base.py
│   ├── repository_user.py
│   └── repository_product.py
│
├── validators/                    # Все файлы с префиксом "validator"
│   ├── validator_email.py
│   ├── validator_password.py
│   └── validator_phone.py
│
├── utils/                         # Все файлы с префиксом "utils"
│   ├── utils_string.py
│   ├── utils_date.py
│   └── utils_crypto.py
│
├── tests/                         # Все файлы с префиксом "test"
│   ├── test_main.py
│   ├── test_config.py
│   ├── test_model_user.py
│   ├── test_service_auth.py
│   └── conftest.py              # Исключение - конфигурация pytest
│
├── .env                           # Секреты (в .gitignore!)
├── .env.example
├── .gitignore
├── main.log
├── requirements.txt
└── pyproject.toml
```
**Передача файлов ассистенту:**
```
main.py, config.py, app.py, model_base.py, model_user.py, 
service_auth.py, repository_user.py, validator_email.py, 
utils_string.py, test_main.py
```
Ассистент сразу понимает структуру без путей!
---
## РАСШИРЕННЫЕ ПРИМЕРЫ FILE_NAME_IMMUTABILITY
### Сценарии модификации
**Сценарий 1: Рефакторинг внутри файла**
```python
# user_manager.py - ДО рефакторинга (18 Кб)
class UserManager:
def create_user(self): ...
def validate_user(self): ...
def save_user(self): ...
# ... много методов ...
# user_manager.py - ПОСЛЕ рефакторинга (18 Кб)
# Имя файла НЕ ИЗМЕНИЛОСЬ!
class UserManager:
def __init__(self):
self.validator = UserValidator()
self.repository = UserRepository()
def create_user(self): ...
class UserValidator: # Выделен в класс внутри ТОГО ЖЕ файла
def validate_user(self): ...
class UserRepository: # Выделен в класс внутри ТОГО ЖЕ файла
def save_user(self): ...
```
**Сценарий 2: Файл превышает 20 Кб**
```
Ассистент: "Файл user_manager.py достиг 22 Кб.
Рекомендую разделение:
Вариант A (рекомендуемый):
• user_manager.py (8 Кб) - основная логика
• user_validator.py (6 Кб) - валидация
• user_repository.py (8 Кб) - БД операции
Вариант B:
• Оставить user_manager.py, упростить код
Какой вариант выбираете?"
Пользователь: "Вариант A"
Ассистент: "Создаю новые файлы:
✓ user_validator.py
✓ user_repository.py
✓ Обновляю user_manager.py (импорты + основная логика)"
```
**Сценарий 3: Пользователь просит переименование**
```
Пользователь: "Переименуй user_manager.py в user_service.py"
Ассистент: "Переименование выполнено:
• Было: user_manager.py
• Стало: user_service.py
Обновлены импорты в файлах:
• main.py
• test_user_manager.py → test_user_service.py"
```
---
## РАСШИРЕННЫЕ ПРИМЕРЫ LOGGING
### Детальная конфигурация
```python
import logging
import sys
from pathlib import Path
# Продвинутая конфигурация
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(
level=logging.DEBUG,
format=LOG_FORMAT,
datefmt=DATE_FORMAT,
handlers=[
logging.FileHandler('main.log', encoding='utf-8', mode='a'),
logging.StreamHandler(sys.stdout)
]
)
# Создание логгера для модуля
logger = logging.getLogger(__name__)
# Опциональ��о: отключение логов библиотек
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
```
### Паттерны логирования в различных ситуациях
**1. Логирование функции с параметрами:**
```python
def process_payment(user_id: int, amount: float, currency: str) -> dict:
"""Process payment for user."""
logger.info(f"Processing payment: user_id={user_id}, amount={amount}, currency={currency}")
try:
# Валидация
if amount <= 0:
logger.warning(f"Invalid amount: {amount} for user {user_id}")
raise ValueError("Amount must be positive")
# Обработка
logger.debug(f"Calling payment gateway for user {user_id}")
result = payment_gateway.charge(user_id, amount, currency)
logger.info(f"Payment successful: transaction_id={result['transaction_id']}")
return result
except PaymentGatewayError as e:
logger.error(f"Payment gateway error for user {user_id}: {e}", exc_info=True)
raise
except Exception as e:
logger.exception(f"Unexpected error processing payment for user {user_id}: {e}")
raise
```
**2. Логирование длительных операций:**
```python
import time
def process_large_dataset(data: list) -> dict:
"""Process large dataset with progress logging."""
start_time = time.time()
total_items = len(data)
logger.info(f"Starting dataset processing: {total_items} items")
results = []
for i, item in enumerate(data):
try:
result = process_item(item)
results.append(result)
# Прогресс каждые 10%
if (i + 1) % (total_items // 10) == 0:
progress = ((i + 1) / total_items) * 100
logger.debug(f"Progress: {progress:.1f}% ({i + 1}/{total_items})")
except Exception as e:
logger.error(f"Failed to process item {i}: {e}", exc_info=True)
continue
elapsed = time.time() - start_time
logger.info(f"Dataset processing completed: {len(results)}/{total_items} successful in {elapsed:.2f}s")
return {"processed": len(results), "failed": total_items - len(results)}
```
**3. Логирование async операций:**
```python
import asyncio
import logging
logger = logging.getLogger(__name__)
async def fetch_multiple_sources(sources: list[str]) -> list[dict]:
"""Fetch data from multiple sources concurrently."""
logger.info(f"Starting concurrent fetch from {len(sources)} sources")
tasks = [fetch_source(source) for source in sources]
try:
results = await asyncio.gather(*tasks, return_exceptions=True)
successful = sum(1 for r in results if not isinstance(r, Exception))
failed = len(results) - successful
logger.info(f"Fetch completed: {successful} successful, {failed} failed")
return [r for r in results if not isinstance(r, Exception)]
except Exception as e:
logger.exception(f"Critical error in concurrent fetch: {e}")
raise
async def fetch_source(source: str) -> dict:
"""Fetch data from single source."""
logger.debug(f"Fetching from {source}")
try:
await asyncio.sleep(1) # Simulate I/O
logger.debug(f"Successfully fetched from {source}")
return {"source": source, "data": "..."}
except Exception as e:
logger.error(f"Failed to fetch from {source}: {e}", exc_info=True)
raise
```
**4. Логирование в main() точке входа:**
```python
def main():
"""Main application entry point."""
logger.info("="*50)
logger.info("Application starting")
logger.info("="*50)
try:
# Загрузка конфигурации
logger.debug("Loading configuration")
config = load_config()
logger.info(f"Configuration loaded: {config.get('environment')}")
# Инициализация компонентов
logger.debug("Initializing components")
initialize_components(config)
logger.info("Components initialized successfully")
# Запуск приложения
logger.info("Starting main application loop")
run_application()
except ConfigurationError as e:
logger.critical(f"Configuration error: {e}", exc_info=True)
sys.exit(1)
except Exception as e:
logger.critical(f"Fatal error: {e}", exc_info=True)
sys.exit(1)
finally:
logger.info("="*50)
logger.info("Application shutdown")
logger.info("="*50)
if __name__ == "__main__":
# Настройка логирования
logging.basicConfig(
level=logging.DEBUG,
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
handlers=[
logging.FileHandler('main.log', encoding='utf-8'),
logging.StreamHandler(sys.stdout)
]
)
main()
```
---
## РАСШИРЕННЫЕ ПРИМЕРЫ TYPE_HINTS
### Сложные типы
```python
from typing import TypeVar, Generic, Protocol, Callable, Union, Optional, Any
from collections.abc import Iterable, Sequence, Mapping
# Generic классы
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
class Repository(Generic[T]):
"""Generic repository pattern."""
def get(self, id: int) -> Optional[T]:
...
def get_all(self) -> list[T]:
...
def save(self, entity: T) -> T:
...
# Protocol для structural typing
class Drawable(Protocol):
def draw(self) -> None:
...
def render(item: Drawable) -> None:
item.draw()
# Callable types
Validator = Callable[[str], bool]
Transformer = Callable[[T], T]
def apply_validators(value: str, validators: list[Validator]) -> bool:
return all(validator(value) for validator in validators)
# Union types (Python 3.10+)
def process_id(id: int | str) -> str:
return str(id)
# Nested generic types
UserCache = dict[int, Optional[tuple[str, str]]]
def get_user_cache() -> UserCache:
return {1: ("John", "Doe"), 2: None}
# Type aliases
JsonDict = dict[str, Any]
Headers = Mapping[str, str]
def parse_json(data: str) -> JsonDict:
...
```
---
## РАСШИРЕННЫЕ ПРИМЕРЫ DOCSTRINGS
### NumPy style (детальный)
```python
def fetch_user_data(
user_id: int,
include_orders: bool = False,
max_orders: int = 10,
timeout: float = 30.0
) -> dict[str, Any]:
"""Fetch user data from database with optional order history.
This function retrieves user information and optionally includes
order history. Network timeout can be customized.
Parameters
----------
user_id : int
Unique identifier for the user
include_orders : bool, optional
Whether to include order history (default: False)
max_orders : int, optional
Maximum number of orders to retrieve (default: 10)
Only used when include_orders is True
timeout : float, optional
Network timeout in seconds (default: 30.0)
Returns
-------
dict[str, Any]
Dictionary containing user data with keys:
- 'id': int - User ID
- 'name': str - User full name
- 'email': str - User email
- 'orders': list[dict] - Order history (if include_orders=True)
Raises
------
ValueError
If user_id is negative or max_orders < 1
DatabaseConnectionError
If database connection fails
TimeoutError
If request exceeds timeout duration
UserNotFoundError
If user with given ID doesn't exist
See Also
--------
create_user : Create new user in database
update_user : Update existing user data
Notes
-----
This function uses connection pooling for better performance.
Large max_orders values (>100) may impact performance.
Examples
--------
>>> user = fetch_user_data(123)
>>> print(user['name'])
'John Doe'
>>> user_with_orders = fetch_user_data(123, include_orders=True, max_orders=5)
>>> len(user_with_orders['orders'])
5
>>> # With custom timeout
>>> user = fetch_user_data(123, timeout=60.0)
"""
logger.debug(f"Fetching user data: user_id={user_id}")
# ... implementation ...
```
### Google style (альтернатива)
```python
def process_payment(
amount: float,
currency: str,
card_token: str,
description: Optional[str] = None
) -> dict[str, Any]:
"""Process payment transaction.
Args:
amount: Payment amount (must be positive)
currency: Three-letter currency code (e.g., 'USD', 'EUR')
card_token: Tokenized card information
description: Optional payment description
Returns:
Dictionary with transaction details:
{
'transaction_id': str,
'status': str,
'timestamp': datetime
}
Raises:
ValueError: If amount <= 0 or invalid currency code
PaymentError: If payment processing fails
Example:
>>> result = process_payment(100.0, 'USD', 'tok_12345')
>>> result['status']
'success'
"""
pass
```
---
## SOLID ПРИНЦИПЫ (расширенные примеры)
### Single Responsibility Principle
```python
# ❌ ПЛОХО - класс делает слишком много
class User:
def __init__(self, name: str, email: str):
self.name = name
self.email = email
def save_to_db(self):
# БД логика
pass
def send_welcome_email(self):
# Email логика
pass
def generate_pdf_report(self):
# PDF логика
pass
def validate_email(self):
# Валидация
pass
# ✅ ХОРОШО - разделение ответственности
class User:
"""Представляет пользователя (только данные)."""
def __init__(self, name: str, email: str):
self.name = name
self.email = email
class UserRepository:
"""Отвечает за сохранение/загрузку."""
def save(self, user: User) -> None:
pass
def find_by_id(self, user_id: int) -> Optional[User]:
pass
class EmailService:
"""Отвечает за отправку email."""
def send_welcome(self, user: User) -> None:
pass
class ReportGenerator:
"""Отвечает за генерацию отчетов."""
def generate_user_report(self, user: User) -> bytes:
pass
class UserValidator:
"""Отвечает за валидацию."""
def validate_email(self, email: str) -> bool:
pass
```
### Open-Closed Principle
```python
from abc import ABC, abstractmethod
# ✅ Открыт для расширения, закрыт для модификации
class NotificationService(ABC):
@abstractmethod
def send(self, recipient: str, message: str) -> bool:
"""Send notification."""
pass
class EmailNotification(NotificationService):
def send(self, recipient: str, message: str) -> bool:
logger.info(f"Sending email to {recipient}")
# Email implementation
return True
class SMSNotification(NotificationService):
def send(self, recipient: str, message: str) -> bool:
logger.info(f"Sending SMS to {recipient}")
# SMS implementation
return True
class PushNotification(NotificationService):
def send(self, recipient: str, message: str) -> bool:
logger.info(f"Sending push to {recipient}")
# Push implementation
return True
# Использование
def notify_user(service: NotificationService, recipient: str, message: str):
"""Работает с любым типом уведомлений."""
return service.send(recipient, message)
# Добавление нового типа не требует изменения существующего кода
class TelegramNotification(NotificationService):
def send(self, recipient: str, message: str) -> bool:
logger.info(f"Sending Telegram message to {recipient}")
return True
```
### Dependency Inversion Principle
```python
from abc import ABC, abstractmethod
# ❌ ПЛОХО - зависимость от конкретной реализации
class MySQLDatabase:
def execute_query(self, query: str):
pass
class UserService:
def __init__(self):
self.db = MySQLDatabase() # Жесткая зависимость!
def get_user(self, user_id: int):
return self.db.execute_query(f"SELECT * FROM users WHERE id={user_id}")
# ✅ ХОРОШО - зависимость от абстракции
class Database(ABC):
@abstractmethod
def execute_query(self, query: str):
pass
class MySQLDatabase(Database):
def execute_query(self, query: str):
logger.debug(f"Executing MySQL query: {query}")
# MySQL implementation
pass
class PostgreSQLDatabase(Database):
def execute_query(self, query: str):
logger.debug(f"Executing PostgreSQL query: {query}")
# PostgreSQL implementation
pass
class UserService:
def __init__(self, database: Database):
self.db = database # Зависимость от абстракции
def get_user(self, user_id: int):
return self.db.execute_query(f"SELECT * FROM users WHERE id={user_id}")
# Использование
mysql_db = MySQLDatabase()
user_service = UserService(mysql_db)
# Легко заменить на другую БД
postgres_db = PostgreSQLDatabase()
user_service = UserService(postgres_db)
```
---
## TESTING (расширенные примеры)
### Fixtures и параметризация
```python
import pytest
from typing import Generator
# conftest.py
@pytest.fixture(scope="session")
def database_connection() -> Generator:
"""Создание БД соединения для всех тестов."""
logger.info("Setting up database connection")
conn = create_database_connection()
yield conn
logger.info("Closing database connection")
conn.close()
@pytest.fixture(scope="function")
def clean_database(database_connection) -> None:
"""Очистка БД перед каждым тестом."""
logger.debug("Cleaning database")
database_connection.execute("DELETE FROM users")
database_connection.commit()
@pytest.fixture
def sample_user() -> dict:
"""Создание тестового пользователя."""
return {
"name": "Test User",
"email": "test@example.com",
"age": 25
}
# test_user_service.py
def test_create_user_success(clean_database, sample_user):
# Arrange
service = UserService(database_connection)
# Act
result = service.create_user(sample_user)
# Assert
assert result["id"] is not None
assert result["name"] == sample_user["name"]
assert result["email"] == sample_user["email"]
@pytest.mark.parametrize("invalid_email", [
"",
"invalid",
"missing@domain",
"@nodomain.com",
"no_at_sign.com"
])
def test_create_user_invalid_email(invalid_email, clean_database):
# Arrange
service = UserService(database_connection)
user_data = {"name": "Test", "email": invalid_email}
# Act & Assert
with pytest.raises(ValidationError) as exc_info:
service.create_user(user_data)
assert "email" in str(exc_info.value).lower()
@pytest.mark.parametrize("age,expected_category", [
(5, "child"),
(15, "teenager"),
(25, "adult"),
(70, "senior"),
])
def test_age_categorization(age, expected_category):
# Act
result = categorize_by_age(age)
# Assert
assert result == expected_category
```
---
## ASYNC ПАТТЕРНЫ (расширенные)
```python
import asyncio
from typing import Any
import aiohttp
async def fetch_with_retry(
url: str,
max_retries: int = 3,
timeout: float = 10.0
) -> dict[str, Any]:
"""Fetch data with automatic retry logic."""
for attempt in range(max_retries):
try:
logger.debug(f"Attempt {attempt + 1}/{max_retries} for {url}")
async with aiohttp.ClientSession() as session:
async with session.get(url, timeout=timeout) as response:
response.raise_for_status()
data = await response.json()
logger.info(f"Successfully fetched {url}")
return data
except asyncio.TimeoutError:
logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
if attempt == max_retries - 1:
raise
await asyncio.sleep(2 ** attempt) # Exponential backoff
except aiohttp.ClientError as e:
logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
if attempt == max_retries - 1:
raise
await asyncio.sleep(2 ** attempt)
async def process_batch(items: list[str], batch_size: int = 10) -> list[dict]:
"""Process items in batches to avoid overwhelming resources."""
results = []
for i in range(0, len(items), batch_size):
batch = items[i:i + batch_size]
logger.debug(f"Processing batch {i // batch_size + 1}")
batch_results = await asyncio.gather(
*[fetch_with_retry(item) for item in batch],
return_exceptions=True
)
results.extend(batch_results)
# Small delay between batches
if i + batch_size < len(items):
await asyncio.sleep(0.1)
return results
```