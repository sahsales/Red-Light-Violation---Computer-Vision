# Importações
import cv2
from ultralytics import YOLO

# Carregar modelo e configuração do vídeo

modelo = YOLO('best.pt')

video_path = 'infracaoTest.mp4'
video_saida = 'resultado4.mp4'

cap = cv2.VideoCapture(video_path)

largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30

out = cv2.VideoWriter(video_saida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))


# Tradução das classes para o português

traducao_classes = {
    "veiculo": "Veiculo",
    "red_light": "Semaforo Vermelho",
    "green_light": "Semaforo Verde"
}

classes_veiculos = ["veiculo"]

infracoes = set()
veiculos_ja_cruzaram = set()  # Registra veículos que já cruzaram a linha (seja no verde ou vermelho)
posicoes_anteriores = {}      # Guarda o centro da base do veículo no frame anterior {id: (x, y)}

frames_desde_vermelho = 999
limite_vermelho = 25

# Desenhar caixa de texto com fundo colorido

def desenhar_texto_com_fundo(img, texto, pos, cor_fundo, cor_texto=(0, 0, 0), escala=0.35, espessura=1):
    x, y = pos
    (largura_txt, altura_txt), baseline = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, espessura)
    cv2.rectangle(img, (x, y - altura_txt - 4), (x + largura_txt + 4, y + baseline), cor_fundo, -1)
    cv2.putText(img, texto, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, escala, cor_texto, espessura, cv2.LINE_AA)


# Interação com a linha

ponto_inicial = None
ponto_final = None
desenhando = False
linha_definida = False

def desenhar_linha(event, x, y, flags, param):
    global ponto_inicial, ponto_final, desenhando, linha_definida
    
    if event == cv2.EVENT_LBUTTONDOWN:
        ponto_inicial = (x, y)
        ponto_final = (x, y)
        desenhando = True
        linha_definida = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if desenhando:
            ponto_final = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        ponto_final = (x, y)
        desenhando = False
        linha_definida = True

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segmentos_cruzam(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def caixa_intercepta_linha(box_coords, p1, p2, id_veiculo):
    x1, y1, x2, y2 = box_coords
    
    # Checagem direta no frame atual (linha inferior da bounding box)
    borda_inferior_inicio = (x1, y2)
    borda_inferior_fim = (x2, y2)
    if segmentos_cruzam(borda_inferior_inicio, borda_inferior_fim, p1, p2):
        return True

    # Checagem de deslocamento 

    centro_base_atual = ((x1 + x2) // 2, y2)
    if id_veiculo in posicoes_anteriores:
        centro_base_anterior = posicoes_anteriores[id_veiculo]
        if segmentos_cruzam(centro_base_anterior, centro_base_atual, p1, p2):
            return True

    # Atualiza a posição da base para a próxima checagem

    posicoes_anteriores[id_veiculo] = centro_base_atual
    return False

# Capturar primeiro frame para desenho

ret, primeiro_frame = cap.read()
if not ret:
    print("Erro ao carregar o vídeo.")
    exit()

cv2.namedWindow("Desenhar Linha de Parada", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Desenhar Linha de Parada", 800, 600)
cv2.setMouseCallback("Desenhar Linha de Parada", desenhar_linha)

print(">>> Desenhe a linha e pressione ENTER ou ESPAÇO para confirmar. <<<")

while True:
    frame_temp = primeiro_frame.copy()
    
    if ponto_inicial and ponto_final:
        cv2.line(frame_temp, ponto_inicial, ponto_final, (0, 255, 255), 2)
        cv2.circle(frame_temp, ponto_inicial, 4, (0, 255, 0), -1)
        cv2.circle(frame_temp, ponto_final, 4, (0, 255, 0), -1)

    cv2.putText(frame_temp, "Desenhe a linha e pressione ENTER para confirmar.", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Desenhar Linha de Parada", frame_temp)
    key = cv2.waitKey(1) & 0xFF
    
    if key in (13, 10, 32) and linha_definida:
        break

cv2.destroyAllWindows()

# Reiniciar captura de vídeo

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cv2.namedWindow("Processamento em Tempo Real", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Processamento em Tempo Real", 800, 600)

# Processar vídeo

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    resultados = modelo.track(
        frame,
        imgsz=832,
        conf=0.25,
        iou=0.5,
        device='cpu',
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    resultado = resultados[0]

    # Detecção de semáforos 

    detectou_vermelho = False

    for box in resultado.boxes:
        classe_id = int(box.cls[0])
        nome_classe = resultado.names[classe_id]
        confianca = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if nome_classe == "red_light":
            detectou_vermelho = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            desenhar_texto_com_fundo(frame, f"Semaforo: VERMELHO {confianca:.2f}", (x1, max(y1, 15)), (0, 0, 255))

        elif nome_classe == "green_light":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            desenhar_texto_com_fundo(frame, f"Semaforo: VERDE {confianca:.2f}", (x1, max(y1, 15)), (0, 255, 0))

    # Desenhar linha manualmente

    if linha_definida:
        cv2.line(frame, ponto_inicial, ponto_final, (0, 255, 255), 2)
        cv2.circle(frame, ponto_inicial, 4, (0, 255, 0), -1)
        cv2.circle(frame, ponto_final, 4, (0, 255, 0), -1)

    # Mémoria do semáforo vermelho 
    
    if detectou_vermelho:
        frames_desde_vermelho = 0
    else:
        frames_desde_vermelho += 1

    semaforo_vermelho = (frames_desde_vermelho <= limite_vermelho)

    # Detecção e interseção de veículos 

    for box in resultado.boxes:

        classe_id = int(box.cls[0])
        nome_classe = resultado.names[classe_id]

        if nome_classe not in classes_veiculos or box.id is None:
            continue

        id_veiculo = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confianca = float(box.conf[0])

        # Checar interseção no frame atual 

        interseccao_ocorreu = False
        if linha_definida:
            interseccao_ocorreu = caixa_intercepta_linha((x1, y1, x2, y2), ponto_inicial, ponto_final, id_veiculo)

        # Registrar cruzamento e validar infração apenas na primeira vez
        if interseccao_ocorreu and id_veiculo not in veiculos_ja_cruzaram:
            veiculos_ja_cruzaram.add(id_veiculo)  # Marca que o veículo já passou pela linha
            
            # Se no momento em que ele cruzou pela PRIMEIRA VEZ o sinal estava vermelho, é infração
            if semaforo_vermelho:
                infracoes.add(id_veiculo)

        # Colocar a tradução

        classe_pt = traducao_classes.get(nome_classe, nome_classe)

        # Colocar cor e texto

        if id_veiculo in infracoes:
            cor = (0, 0, 255)  # Vermelho
            texto = f"{classe_pt} - INFRACAO"
        else:
            cor = (0, 255, 0)  # Verde
            texto = f"{classe_pt} {confianca:.2f}"

        # Desenhar caixa

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 1)
        desenhar_texto_com_fundo(frame, texto, (x1, max(y1, 15)), cor)

    # Estado do semáforo

    texto_semaforo = "Semaforo: VERMELHO" if semaforo_vermelho else "Semaforo: NAO VERMELHO"
    cor_semaforo = (0, 0, 255) if semaforo_vermelho else (0, 255, 0)
    desenhar_texto_com_fundo(frame, texto_semaforo, (20, 30), cor_semaforo, escala=0.35, espessura=1)

    # Mostrar e salvar 

    cv2.imshow("Processamento em Tempo Real", frame)
    
    key = cv2.waitKey(1) & 0xFF
    janela_aberta = cv2.getWindowProperty("Processamento em Tempo Real", cv2.WND_PROP_VISIBLE)
    
    if key == ord('q') or janela_aberta < 1:
        print("Processamento interrompido pelo usuário.")
        break

    out.write(frame)

# Finalizar

cap.release()
out.release()
cv2.destroyAllWindows()

print("Inferência concluída!")
print(f"Vídeo salvo em: {video_saida}")