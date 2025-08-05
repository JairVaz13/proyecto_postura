import os
import cv2

# Configuración
posturas = ['recto', 'deslizado', 'encorvado', 'cruzado', 'inclinado']
for postura in posturas:
    os.makedirs(f'dataset/{postura}', exist_ok=True)

def capturar_imagenes():
    cap = cv2.VideoCapture(0)
    contador = {postura: 0 for postura in posturas}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mostrar instrucciones
        texto = "1:Recto | 2:Deslizado | 3:Encorvado | 4:Cruzado | 5:Inclinado | ESC:Salir"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar contador actual
        contador_texto = " | ".join([f"{p}:{contador[p]}" for p in posturas])
        cv2.putText(frame, contador_texto, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Captura de Posturas', frame)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif 49 <= key <= 53:  # Teclas 1-5
            clase = posturas[key - 49]
            cv2.imwrite(f'dataset/{clase}/postura_{contador[clase]}.jpg', frame)
            contador[clase] += 1
            print(f"Imagen guardada: {clase} ({contador[clase]})")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nResumen final:")
    for p in posturas:
        print(f"{p}: {contador[p]} imágenes")

if __name__ == "__main__":
    capturar_imagenes()