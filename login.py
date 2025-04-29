# -*- coding: utf-8 -*-
import json
import os
import requests
from common.log import logger

CONFIG_FILE = 'config.json'

class LoginHandler:
    def __init__(self, config):
        try:
            self.base_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
                'accept': '*/*',
                'accept-language': 'zh-CN,zh;q=0.9',
                'canary': 'false',
                'connect-protocol-version': '1',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Origin': 'https://yuewen.cn',
                'Pragma': 'no-cache',
                'Referer': 'https://yuewen.cn/chats/new',
                'sec-ch-ua': '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'x-waf-client-type': 'fetch_sdk',
                'priority': 'u=1, i',
                'oasis-appid': '10200',
                'oasis-mode': '2',
                'oasis-platform': 'web'
            }
            self.config = config
            self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE)
        except Exception as e:
            logger.error(f"[Yuewen] LoginHandler初始化失败: {str(e)}")
            raise e

    def save_config(self):
        """保存配置到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # 保存配置
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            
            logger.info(f"[Yuewen] 配置已保存到 {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"[Yuewen] 保存配置失败: {e}")
            return False

    def register_device(self):
        """注册设备获取初始 token"""
        url = 'https://yuewen.cn/passport/proto.api.passport.v1.PassportService/RegisterDevice'
        headers = self.base_headers.copy()
        headers.update({
            'Content-Type': 'application/json',
            'oasis-mode': '1',
            'oasis-webid': '8e2223012fadbac04d9cc1fcdc1d8b4eb8cc75a9'
        })
      
        try:
            response = requests.post(url, headers=headers, json={})
            if response.status_code == 200:
                data = response.json()
                self.config.update({
                    'oasis_webid': data['device']['deviceID'],
                    'oasis_token': f"{data['accessToken']['raw']}...{data['refreshToken']['raw']}"
                })
                self.save_config()
                logger.info(f"[Yuewen] 设备注册成功: {self.config['oasis_webid']}")
                return True
            raise Exception(f"设备注册失败: {response.text}")
        except Exception as e:
            logger.error(f"[Yuewen] 设备注册失败: {e}")
            return False

    def send_verify_code(self, mobile_num):
        url = 'https://yuewen.cn/passport/proto.api.passport.v1.PassportService/SendVerifyCode'
        headers = self.base_headers.copy()
        headers.update({
            'Content-Type': 'application/json',
            'oasis-mode': '1',
            'oasis-webid': self.config['oasis_webid']
        })
      
        cookies = {
            'Oasis-Webid': self.config['oasis_webid'],
            'Oasis-Token': self.config['oasis_token']
        }
      
        data = {'mobileCc': '86', 'mobileNum': mobile_num}
        response = requests.post(url, headers=headers, cookies=cookies, json=data)
        return response.status_code == 200

    def sign_in(self, mobile_num, auth_code):
        url = 'https://yuewen.cn/passport/proto.api.passport.v1.PassportService/SignIn'
        headers = self.base_headers.copy()
        headers.update({
            'Content-Type': 'application/json',
            'oasis-mode': '1',
            'oasis-webid': self.config['oasis_webid']
        })
      
        cookies = {
            'Oasis-Webid': self.config['oasis_webid'],
            'Oasis-Token': self.config['oasis_token']
        }
      
        data = {
            'authCode': auth_code,
            'mobileCc': '86',
            'mobileNum': mobile_num
        }
      
        response = requests.post(url, headers=headers, cookies=cookies, json=data)
        if response.status_code == 200:
            data = response.json()
            self.config['oasis_token'] = f"{data['accessToken']['raw']}...{data['refreshToken']['raw']}"
            self.save_config()
            return True
        return False

    def verify_login(self, mobile_num, verify_code):
        """验证码登录（sign_in的别名）"""
        return self.sign_in(mobile_num, verify_code)

    def refresh_token(self):
        """刷新令牌"""
        logger.info("[Yuewen] 检测到令牌过期，尝试刷新令牌...")
        url = 'https://yuewen.cn/passport/proto.api.passport.v1.PassportService/RefreshToken'
        headers = self.base_headers.copy()
        headers.update({
            'Content-Type': 'application/json',
            'oasis-mode': '1',
            'oasis-webid': self.config['oasis_webid'],
            'Cookie': f"Oasis-Webid={self.config['oasis_webid']}; Oasis-Token={self.config['oasis_token']}"
        })
      
        try:
            response = requests.post(url, headers=headers, json={})
            if response.status_code == 200:
                data = response.json()
                self.config['oasis_token'] = f"{data['accessToken']['raw']}...{data['refreshToken']['raw']}"
                self.save_config()
                logger.info("[Yuewen] 令牌刷新成功！")
                return True
            else:
                logger.error(f"[Yuewen] 刷新失败: {response.text}")
                return False
        except Exception as e:
            logger.error(f"[Yuewen] 刷新请求异常: {e}")
            return False

    def login_flow(self):
        if not self.config.get('oasis_webid'):
            self.register_device()
      
        mobile_num = input("请输入11位手机号码: ").strip()
        while len(mobile_num) != 11 or not mobile_num.isdigit():
            mobile_num = input("无效的号码，请重新输入: ").strip()

        print("发送验证码中...")
        if self.send_verify_code(mobile_num):
            auth_code = input("请输入4位验证码: ").strip()
            while len(auth_code) != 4 or not auth_code.isdigit():
                auth_code = input("无效的验证码，请重新输入: ").strip()

            if self.sign_in(mobile_num, auth_code):
                print("登录成功！")
                self.config['need_login'] = False
                self.save_config()
                return True
            else:
                print("登录失败")
                return False
        return False