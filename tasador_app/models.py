from django.contrib.gis.db import models
from django.contrib.gis.geos import Point
from django.contrib.auth.models import User

DISTRICT_CHOICES_DEPARTAMENTOS = (
    ('Ancon', 'Ancon'), ('Ate', 'Ate'), ('Barranco', 'Barranco'), ('Bellavista', 'Bellavista'), ('Breña', 'Breña'),
    ('Callao', 'Callao'), ('Carabayllo', 'Carabayllo'), ('Cercado de Lima', 'Cercado de Lima'),
    ('Chaclacayo', 'Chaclacayo'), ('Chorrillos', 'Chorrillos'), ('Chosica', 'Chosica'), ('Comas', 'Comas'),
    ('El Agustino', 'El Agustino'), ('Jesus Maria', 'Jesus Maria'), ('La Molina', 'La Molina'), ('La Perla', 'La Perla'),
    ('La Punta', 'La Punta'), ('La Victoria', 'La Victoria'), ('Lince', 'Lince'), ('Los Olivos', 'Los Olivos'),
    ('Lurin', 'Lurin'), ('Magdalena del Mar', 'Magdalena del Mar'), ('Miraflores', 'Miraflores'),
    ('Pachacamac', 'Pachacamac'), ('Pueblo Libre', 'Pueblo Libre'), ('Puente Piedra', 'Puente Piedra'),
    ('Punta Hermosa', 'Punta Hermosa'), ('Rimac', 'Rimac'), ('San Bartolo', 'San Bartolo'), ('San Borja', 'San Borja'),
    ('San Isidro', 'San Isidro'), ('San Juan de Lurigancho', 'San Juan de Lurigancho'),
    ('San Juan de Miraflores', 'San Juan de Miraflores'), ('San Luis', 'San Luis'),
    ('San Martin de Porres', 'San Martin de Porres'), ('San Miguel', 'San Miguel'), ('Santa Anita', 'Santa Anita'),
    ('Santa Maria del Mar', 'Santa Maria del Mar'), ('Santa Rosa', 'Santa Rosa'),
    ('Santiago de Surco', 'Santiago de Surco'), ('Surquillo', 'Surquillo'), ('Ventanilla', 'Ventanilla'),
    ('Villa El Salvador', 'Villa El Salvador')
)

class Departamento(models.Model):
    distrito = models.CharField(max_length=100, choices=DISTRICT_CHOICES_DEPARTAMENTOS, default='Escoge distrito')
    area = models.FloatField()
    dormitorios = models.IntegerField()
    baños = models.IntegerField()
    ubicacion = models.PointField(default=Point( -77.0282, -12.0432, srid=4326))
    empresario = models.ForeignKey(User, default=None, on_delete='CASCADE')

DISTRICT_CHOICES_CASAS = (
    ('Ancon', 'Ancon'), ('Ate', 'Ate'), ('Barranco', 'Barranco'), ('Bellavista', 'Bellavista'), ('Breña', 'Breña'),
    ('Callao', 'Callao'), ('Carabayllo', 'Carabayllo'), ('Cercado de Lima', 'Cercado de Lima'),
    ('Chaclacayo', 'Chaclacayo'), ('Chorrillos', 'Chorrillos'), ('Chosica', 'Chosica'), ('Cieneguilla', 'Cieneguilla'),
    ('Comas', 'Comas'), ('El Agustino', 'El Agustino'), ('Independencia', 'Independencia'),
    ('Jesus Maria', 'Jesus Maria'), ('La Molina', 'La Molina'), ('La Perla', 'La Perla'), ('La Punta', 'La Punta'),
    ('La Victoria', 'La Victoria'), ('Lince', 'Lince'), ('Los Olivos', 'Los Olivos'), ('Lurin', 'Lurin'),
    ('Magdalena del Mar', 'Magdalena del Mar'), ('Miraflores', 'Miraflores'), ('Pachacamac', 'Pachacamac'), ('Pucusana', 'Pucusana'),
    ('Pueblo Libre', 'Pueblo Libre'), ('Puente Piedra', 'Puente Piedra'), ('Punta Hermosa', 'Punta Hermosa'),
    ('Punta Negra', 'Punta Negra'), ('Rimac', 'Rimac'),  ('San Bartolo', 'San Bartolo'), ('San Borja', 'San Borja'),
    ('San Isidro', 'San Isidro'),  ('San Juan de Lurigancho', 'San Juan de Lurigancho'),
    ('San Juan de Miraflores', 'San Juan de Miraflores'), ('San Luis', 'San Luis'),
    ('San Martin de Porres', 'San Martin de Porres'), ('San Miguel', 'San Miguel'), ('Santa Anita', 'Santa Anita'),
    ('Santa Maria del Mar', 'Santa Maria del Mar'), ('Santa Rosa', 'Santa Rosa'),
    ('Santiago de Surco', 'Santiago de Surco'), ('Surquillo', 'Surquillo'), ('Ventanilla', 'Ventanilla'),
    ('Villa El Salvador', 'Villa El Salvador'), ('Villa Maria del Triunfo', 'Villa Maria del Triunfo')
)

class Casa(models.Model):
    distrito = models.CharField(max_length=100, choices=DISTRICT_CHOICES_CASAS, default='Escoge distrito')
    area = models.FloatField()
    dormitorios = models.IntegerField()
    baños = models.IntegerField()
    ubicacion = models.PointField(default=Point( -77.0282, -12.0432, srid=4326))
    empresario = models.ForeignKey(User, default=None, on_delete='CASCADE')

DISTRICT_CHOICES_TERRENOS = (
    ('Ancon', 'Ancon'), ('Ate', 'Ate'), ('Barranco', 'Barranco'), ('Bellavista', 'Bellavista'), ('Breña', 'Breña'),
    ('Callao', 'Callao'), ('Carabayllo', 'Carabayllo'), ('Carmen De La Legua Reynoso', 'Carmen De La Legua Reynoso'),
    ('Cercado de Lima', 'Cercado de Lima'),
    ('Chaclacayo', 'Chaclacayo'), ('Chorrillos', 'Chorrillos'), ('Chosica', 'Chosica'), ('Cieneguilla', 'Cieneguilla'),
    ('Comas', 'Comas'), ('El Agustino', 'El Agustino'), ('Independencia', 'Independencia'),
    ('Jesus Maria', 'Jesus Maria'), ('La Molina', 'La Molina'), ('La Perla', 'La Perla'),
    ('La Victoria', 'La Victoria'), ('Lince', 'Lince'), ('Los Olivos', 'Los Olivos'), ('Lurin', 'Lurin'),
    ('Magdalena del Mar', 'Magdalena del Mar'), ('Miraflores', 'Miraflores'), ('Pachacamac', 'Pachacamac'), ('Pucusana', 'Pucusana'),
    ('Pueblo Libre', 'Pueblo Libre'), ('Puente Piedra', 'Puente Piedra'), ('Punta Hermosa', 'Punta Hermosa'),
    ('Punta Negra', 'Punta Negra'), ('Rimac', 'Rimac'),  ('San Bartolo', 'San Bartolo'), ('San Borja', 'San Borja'),
    ('San Isidro', 'San Isidro'),  ('San Juan de Lurigancho', 'San Juan de Lurigancho'),
    ('San Juan de Miraflores', 'San Juan de Miraflores'), ('San Luis', 'San Luis'),
    ('San Martin de Porres', 'San Martin de Porres'), ('San Miguel', 'San Miguel'), ('Santa Anita', 'Santa Anita'),
    ('Santa Maria del Mar', 'Santa Maria del Mar'), ('Santa Rosa', 'Santa Rosa'),
    ('Santiago de Surco', 'Santiago de Surco'), ('Surquillo', 'Surquillo'), ('Ventanilla', 'Ventanilla'),
    ('Villa El Salvador', 'Villa El Salvador'), ('Villa Maria del Triunfo', 'Villa Maria del Triunfo')
)

class Terreno(models.Model):
    distrito = models.CharField(max_length=100, choices=DISTRICT_CHOICES_TERRENOS, default='Escoge distrito')
    area = models.FloatField()
    ubicacion = models.PointField(default=Point(-77.0282, -12.0432, srid=4326))
    empresario = models.ForeignKey(User, default=None, on_delete='CASCADE')