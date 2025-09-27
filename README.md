# Water Monitoring 🌊

<div align="center">

![Django](https://img.shields.io/badge/Django-5.0.14-092E20?style=for-the-badge&logo=django&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![PostGIS](https://img.shields.io/badge/PostGIS-336791?style=for-the-badge&logo=postgresql&logoColor=white)
![Celery](https://img.shields.io/badge/Celery-37B24D?style=for-the-badge&logo=celery&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)

**Aplicación Django para monitoreo de cuerpos de agua**

</div>

---

## 📋 Tabla de contenido

- [🔧 Requisitos previos](#-requisitos-previos)
- [⚡ Instalación](#-instalación)
- [🏗️ Estructura del proyecto](#️-estructura-del-proyecto)
- [🚀 Producción](#-producción)
- [💬 Soporte](#-soporte)

---

## 🔧 Requisitos previos

Antes de comenzar, asegúrate de tener instalados los siguientes componentes:

| Componente | Versión | Enlace |
|------------|---------|--------|
| **Python** | 3.8+ | [python.org](https://python.org) |
| **PostgreSQL** | 12+ | [PostgreSQL](https://postgresql.org) |
| **PostGIS** | Latest | [PostGIS](https://postgis.net) |
| **GDAL** | 3.11.3 | [GISInternals](https://www.gisinternals.com) |

---

## ⚡ Instalación

### 1️⃣ Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
```

### 2️⃣ Configurar entorno virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows
venv\Scripts\activate

# Linux/MacOS
source venv/bin/activate
```

### 3️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Ver dependencias principales</summary>

```txt
Django==5.0.14
psycopg2-binary==2.9.9
djangorestframework==3.15.2
drf-spectacular==0.27.2
django-celery-results==2.5.1
celery==5.4.0
python-dotenv==1.0.1
gdal==3.11.3
djangorestframework-simplejwt==5.3.1
```
</details>

### 4️⃣ Configurar GDAL

GDAL es **esencial** para el soporte geoespacial (`django.contrib.gis`).

<details>
<summary>🪟 <strong>Configuración para Windows</strong></summary>

1. **Descargar GDAL** desde [GISInternals](https://www.gisinternals.com)
2. **Extraer** en `C:\GDAL`
3. **Agregar al PATH**: `C:\GDAL\bin`
   - Configuración → Sistema → Variables de entorno → PATH
4. **Configurar en Django**:
   ```python
   # En settings.py
   GDAL_LIBRARY_PATH = r'C:\GDAL\bin\gdal.dll'
   ```
   O como variable de entorno:
   ```bash
   set GDAL_LIBRARY_PATH=C:\GDAL\bin\gdal.dll
   ```
</details>

<details>
<summary>🐧 <strong>Configuración para Linux/MacOS</strong></summary>

**Ubuntu/Debian:**
```bash
sudo apt-get install gdal-bin libgdal-dev
```

**MacOS (Homebrew):**
```bash
brew install gdal
```

</details>

### 5️⃣ Variables de entorno

Crear archivo `.env` en la raíz del proyecto como el siguiente ejemplo:

```env
SECRET_KEY=django-insecure-hma*-m#@y*iq5*0+)unicm)v((1bqaj)y(_15$to867_3a95m8
DB_NAME=water_monitoring
DB_USER=postgres
DB_PASSWORD=12345678
DB_HOST=localhost
DB_PORT=5432
CELERY_BROKER_URL=django://
```


### 6️⃣ Configurar base de datos

```sql
-- Crear base de datos
CREATE DATABASE water_monitoring;

-- Habilitar PostGIS
\c water_monitoring
CREATE EXTENSION postgis;
```

### 7️⃣ Aplicar migraciones

```bash
python manage.py makemigrations
python manage.py migrate
```

### 8️⃣ Configurar Celery

Para procesamiento asíncrono de tareas:

```bash
# Iniciar worker de Celery
celery -A config worker --loglevel=info
```

> 💡 Para producción, considera usar **Redis** o **RabbitMQ** como broker

### 9️⃣ Ejecutar Servidor

```bash
python manage.py runserver
```

🎉 **¡Listo!** Accede a la aplicación en [http://localhost:8000](http://localhost:8000)

---

## 🏗️ Estructura del proyecto

```
water-monitoring/
├── 📁 config/              # Configuración de Django
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── 📁 apps/
│   ├── 📁 monitoring/      # Funcionalidades de monitoreo
│   └── 📁 alerts/          # Gestión de alertas
├── 📁 media/               # Archivos multimedia
├── 📁 static/              # Archivos estáticos
├── 📄 requirements.txt
├── 📄 .env
└── 📄 manage.py
```

---

## 🚀 Producción

### Lista de verificación para deploy

- [ ] Cambiar `DEBUG = False` en `settings.py`
- [ ] Configurar `ALLOWED_HOSTS`
- [ ] Generar nueva `SECRET_KEY` segura
- [ ] Configurar servidor web (Nginx/Apache)
- [ ] Usar WSGI server (Gunicorn/uWSGI)
- [ ] Configurar SSL/TLS
- [ ] Configurar logs apropiados
- [ ] Backup automático de base de datos

> 📚 **Recursos**: [Checklist de Deploy de Django](https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/)

---

## 💬 Soporte

### 📖 Documentación

- [Django Documentation](https://docs.djangoproject.com/)
- [PostGIS Documentation](https://postgis.net/documentation/)
- [Celery Documentation](https://docs.celeryproject.org/)

<div align="center">

**[⬆️ Volver al inicio](#water-monitoring-)**

</div>
