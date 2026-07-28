import gymnasium as gym
import ale_py
import numpy as np
import cv2
import datetime
import os
import sys
import keyboard

Game = "Seaquest"  # 게임 이름을 상수로 정의

# Gymnasium 1.0.0 이상 버전에서는 외부 환경(ALE)을 명시적으로 등록해야 합니다.
gym.register_envs(ale_py)

from envs.my_atari import Atari

def save_episode(data_list):
    """수집된 에피소드 데이터를 .npz 파일로 저장합니다."""
    if not data_list: return
    
    os.makedirs('demonstrations', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"demonstrations/{Game}_{timestamp}.npz"
    
    # 리스트를 넘파이 배열로 변환
    observations = np.array([d[0] for d in data_list], dtype=np.uint8)
    actions = np.array([d[1] for d in data_list], dtype=np.int32)
    rewards = np.array([d[2] for d in data_list], dtype=np.float32)
    # [수정] RL Value 오염 방지를 위해 변수명을 termination으로 명확히 교체
    terminations = np.array([d[3] for d in data_list], dtype=bool) 
    
    np.savez_compressed(filename, obs=observations, action=actions, reward=rewards, termination=terminations)
    print(f"\n[저장 완료] 에피소드가 {filename}에 저장되었습니다. (길이: {len(data_list)})")

def copy_image_to_clipboard(filepath):
    """지정된 경로의 이미지를 클립보드에 복사합니다 (Windows 전용, 비동기 실행)."""
    if os.name == 'nt':
        import subprocess
        import threading
        def _copy():
            abs_path = os.path.abspath(filepath)
            cmd = [
                "powershell",
                "-command",
                f"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Clipboard]::SetImage([System.Drawing.Image]::FromFile('{abs_path}'))"
            ]
            # 새 콘솔 창이 뜨지 않도록 설정 플래그 사용 (CREATE_NO_WINDOW)
            subprocess.run(cmd, creationflags=0x08000000)
        threading.Thread(target=_copy).start()

def manual_play_with_save(mode="rl", save_res="model", size=84):
    if os.name == 'posix' and sys.platform != 'darwin' and not os.environ.get('DISPLAY'):
        print("\n[오류] 디스플레이(모니터)를 찾을 수 없습니다.")
        print("서버 환경(Headless)에서는 게임 화면을 띄울 수 없습니다.")
        print("이 스크립트를 로컬 PC에서 실행한 후, 생성된 .npz 파일을 서버로 복사해 주세요.\n")
        return

    length = 0 if mode == "ppt" else 108000
    # [수정] 실제 모델 학습과 동일하게 actions="needed"를 사용합니다. 목숨 하나만 잃어도 리셋되도록 합니다.
    env = Atari(name=f'ALE/{Game}-v5', action_repeat=4, size=(size, size), gray=False, actions="needed", length=length, lives="reset")
    obs, info = env.reset()
    
    total_reward = 0
    episode_data = [] 
    
    print("\n" + "="*30)
    if mode == "ppt":
        print(f"{Game} 직접 플레이 & PPT 발표 모드 (시간 제한 없음)")
    else:
        print(f"{Game} 직접 플레이 & 데이터 저장 모드")
    print("조작: WASD(이동), Space(Action/Fire), Enter(일시정지), P(스크린샷), Q(저장 후 종료)")
    print("="*30)

    try:
        while True:
            # 1. 화면 렌더링 (환경 내부 버퍼에 저장된 가장 최근의 고해상도 원본 이미지를 활용)
            hi_res_img = env._buffer[-1] if hasattr(env, '_buffer') else obs
            raw_bgr_img = cv2.cvtColor(hi_res_img, cv2.COLOR_RGB2BGR)
            
            # 게임 플레이 화면용 (전체 여백 유지: 160x210 원본 -> 480x630)
            display_img = cv2.resize(raw_bgr_img, (480, 630), interpolation=cv2.INTER_LINEAR)
            
            # [크롭] 스크린샷용 깔끔한 이미지 생성
            # y축(세로): 상단의 검은 여백(약 26px)과 하단 여백 및 로고(약 176px 이후) 전면 제거
            # x축(가로): 좌우의 검은 여백(각 약 8px) 제거
            cropped_bgr_img = raw_bgr_img[26:210, 8:152]
            display_img_clean = cv2.resize(cropped_bgr_img, (480, 500), interpolation=cv2.INTER_LINEAR)
            
            cv2.putText(display_img, f"Score: {int(total_reward)}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            # 창 이름을 Game으로 변경
            cv2.imshow(f"{Game} - Manual Play & Record", display_img)
            
            # 2. 다중 키 입력 처리 (keyboard 라이브러리 활용)
            cv2.waitKey(50) # 화면 렌더링 유지 및 프레임 속도 조절
            
            action = 0
            w = keyboard.is_pressed('w')
            a = keyboard.is_pressed('a')
            s = keyboard.is_pressed('s')
            d = keyboard.is_pressed('d')
            space = keyboard.is_pressed('space')
            
            if keyboard.is_pressed('q'): 
                save_episode(episode_data)
                break
                
            # 스크린샷 (P 키)
            if keyboard.is_pressed('p'):
                os.makedirs('screenshots', exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                screenshot_path = f"screenshots/screenshot_{timestamp}.png"
                cv2.imwrite(screenshot_path, display_img_clean)
                copy_image_to_clipboard(screenshot_path)
                print(f"[스크린샷 저장 및 클립보드 복사 완료] {screenshot_path}")
                cv2.waitKey(200) # 중복 캡처 방지
                
            # 일시 정지 (Enter 키)
            if keyboard.is_pressed('enter'):
                cv2.putText(display_img, "PAUSED", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow(f"{Game} - Manual Play & Record", display_img)
                cv2.waitKey(300) # 중복 입력 방지 (Debounce)
                while True:
                    if keyboard.is_pressed('enter'):
                        cv2.waitKey(300)
                        break
                    cv2.waitKey(50)
                
            # 복합 키 동적 매핑 (학습 환경의 actions="needed"에 완벽히 호환되도록 구성)
            action_str = ""
            if w: action_str += "UP"
            elif s: action_str += "DOWN"
            if d: action_str += "RIGHT"
            elif a: action_str += "LEFT"
            if space: action_str += "FIRE"
            if not action_str: action_str = "NOOP"

            meanings = env._env.unwrapped.get_action_meanings()
            if action_str in meanings:
                action = meanings.index(action_str)
            elif space and "FIRE" in meanings:
                action = meanings.index("FIRE")
            else:
                direction = action_str.replace("FIRE", "")
                if direction in meanings:
                    action = meanings.index(direction)
                else:
                    action = 0
            
            # 3. 데이터 저장 방식 선택 ('model' = 64x64, 'player' = 원본 화면 해상도)
            save_img = hi_res_img if save_res == "player" else obs

            # 4. 환경 진행
            next_obs, reward, done, next_info = env.step(action)
            total_reward += reward
            
            # [수정] Truncation(시간초과)과 실제 Termination(게임오버)을 분리하여 Value Function 오염 방지
            is_terminal = next_info.get('is_terminal', done)
            episode_data.append((save_img, action, reward, is_terminal))
            obs = next_obs
            info = next_info
            
            if done:
                print(f"Game Over! 최종 점수: {total_reward}")
                save_episode(episode_data)
                obs, info = env.reset()
                total_reward = 0
                episode_data = [] 
                
    finally:
        env.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Private Eye Manual Play")
    parser.add_argument('--mode', type=str, default='rl', choices=['rl', 'ppt'], 
                        help="Mode 'rl' limits length to 108000. Mode 'ppt' has no time limit.")
    parser.add_argument('--save-res', type=str, default='model', choices=['model', 'player'], 
                        help="'model': save 84x84 images. 'player': save original high-res images.")
    parser.add_argument('--size', type=int, default=64, help="resolution of the saved image (default 64 to match training config)")
    args = parser.parse_args()
    
    manual_play_with_save(mode=args.mode, save_res=args.save_res, size=args.size)