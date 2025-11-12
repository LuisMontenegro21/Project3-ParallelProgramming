# Añadimos variables para que fuera mas facil la compilacion de los distintos ejecutables

# Compilador CUDA
NVCC = nvcc

# Archivos fuente y objetos
PGM_SRC = pgm.cpp
PGM_OBJ = pgm.o
PGM_HDR = pgm.h

GLOBAL_SRC = houghGlobal.cu
CONST_SRC  = houghConst.cu
SHARED_SRC = houghShared.cu
LEGACY_SRC = houghBase.cu

# Ejecutables
GLOBAL_EXE = houghGlobal
CONST_EXE  = houghConst
SHARED_EXE = houghShared
LEGACY_EXE = hough_legacy

# make por defecto, compila todas las versiones
all: $(GLOBAL_EXE) $(CONST_EXE) $(SHARED_EXE)

$(GLOBAL_EXE): $(GLOBAL_SRC) $(PGM_OBJ)
	$(NVCC) $(GLOBAL_SRC) $(PGM_OBJ) -o $(GLOBAL_EXE)
	@echo "Compilado: $(GLOBAL_EXE)"

$(CONST_EXE): $(CONST_SRC) $(PGM_OBJ)
	$(NVCC) $(CONST_SRC) $(PGM_OBJ) -o $(CONST_EXE)
	@echo "Compilado: $(CONST_EXE)"

$(SHARED_EXE): $(SHARED_SRC) $(PGM_OBJ)
	$(NVCC) $(SHARED_SRC) $(PGM_OBJ) -o $(SHARED_EXE)
	@echo "Compilado: $(SHARED_EXE)"


# Compilar versiones individuales
global: $(GLOBAL_EXE)
const: $(CONST_EXE)
shared: $(SHARED_EXE)

# Versión completa (legacy)
legacy: $(LEGACY_EXE)

$(LEGACY_EXE): $(LEGACY_SRC) $(PGM_OBJ)
	$(NVCC) $(LEGACY_SRC) $(PGM_OBJ) -o $(LEGACY_EXE)
	@echo "Compilado: $(LEGACY_EXE) (versión de respaldo)"

# Compilación de lector/escritor PGM
$(PGM_OBJ): $(PGM_SRC) $(PGM_HDR)
	g++ -c $(PGM_SRC) -o $(PGM_OBJ)
	@echo "Compilado: $(PGM_OBJ)"

# Limpieza de los ejecutables
clean:
	rm -f $(PGM_OBJ) $(GLOBAL_EXE) $(CONST_EXE) $(SHARED_EXE) $(LEGACY_EXE)
	@echo "Limpieza completa."
