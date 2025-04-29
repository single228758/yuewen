# -*- coding: utf-8 -*-
# yuewen.py
import json
import time
import struct
import random
import os
import httpx 
import re
import requests
import logging
import base64
import numpy as np
import cv2 
from plugins import *
from bridge.context import ContextType, Context
from bridge.reply import Reply, ReplyType
from channel.chat_message import ChatMessage
from common.log import logger
from plugins import Plugin, Event, EventAction, EventContext, register
from config import conf
from .login import LoginHandler

@register(
    name="Yuewen",
    desc="跃问AI助手插件",
    version="0.1",
    author="八戒",
    desire_priority=-1,
    enabled=True
)
class YuewenPlugin(Plugin):
    def get_help_text(self, verbose=False, **kwargs):
        """获取帮助文本"""
        help_text = "跃问AI助手\n"
        if not verbose:
            return help_text
        
        trigger = self.config.get("trigger_prefix", "yw")
        
        help_text += "\n基本指令：\n"
        help_text += f"{trigger} [问题] - 直接对话\n"
        help_text += f"{trigger} 新建会话 - 开启新对话\n\n"
        
        help_text += "模型管理：\n"
        help_text += f"{trigger} 打印模型 - 显示可用模型列表\n"
        help_text += f"{trigger} 更新模型 - 获取最新模型列表\n"
        help_text += f"{trigger} 切换模型 [序号] - 切换到指定模型\n\n"
        
        help_text += "图片功能：\n"
        help_text += f"{trigger} {self.pic_trigger_prefix} - 识别图片内容\n"
        help_text += f"{trigger} {self.pic_trigger_prefix} [提示词] - 根据提示词识别图片\n\n"
        
        help_text += "视频功能：\n"
        help_text += f"{trigger} 视频 [提示词] - 根据提示词生成视频\n"
        help_text += f"{trigger} 视频 [提示词]-润色 - 润色提示词后生成视频\n"
        help_text += f"{trigger} 视频 [提示词]-[镜头语言] - 指定镜头语言生成视频\n"
        help_text += f"{trigger} 参考图 [提示词] - 根据参考图生成视频\n"
        help_text += f"{trigger} 参考图 [提示词]-润色 - 润色提示词后生成视频\n"
        help_text += f"{trigger} 参考图 [提示词]-[镜头语言] - 使用参考图和镜头语言\n\n"
        
        help_text += "其他功能：\n"
        help_text += f"{trigger} 开启联网 - 启用联网模式\n"
        help_text += f"{trigger} 关闭联网 - 关闭联网模式"
        
        return help_text

    def __init__(self):
        super().__init__()
        try:
            # 优先从父类加载配置
            self.config = super().load_config() or {}
            
            # 确保必要的配置项存在
            if "need_login" not in self.config:
                self.config["need_login"] = True
            if "oasis_webid" not in self.config:
                self.config["oasis_webid"] = None
            if "oasis_token" not in self.config:
                self.config["oasis_token"] = None
            if "current_model_id" not in self.config:
                self.config["current_model_id"] = 6
            if "network_mode" not in self.config:
                self.config["network_mode"] = True
            if "trigger_prefix" not in self.config:
                self.config["trigger_prefix"] = "yw"
            if "image_config" not in self.config:
                self.config["image_config"] = {
                    "imgprompt": "解释下图片内容",
                    "trigger": "识图"  # 添加识图触发词配置
                }
                
            # 使用父类方法保存配置
            self.save_config(self.config)
            
            self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
            self.handlers[Event.ON_RECEIVE_MESSAGE] = self.on_receive_message
            self.login_handler = LoginHandler(self.config)
            # 重要：将插件实例传递给login_handler以便保存配置
            self.login_handler._plugin = self
            
            self.client = None
            self.current_chat_id = None
            self.last_active_time = 0
            
            # 添加登录状态管理
            self.is_login_triggered = False
            self.waiting_for_verification = {}  # 等待验证码 {user_id: phone_number}
            
            # 添加模型配置
            self.models = {
                1: {"name": "deepseek r1", "id": 6, "can_network": True},
                2: {"name": "Step2", "id": 2, "can_network": True},
                3: {"name": "Step-R mini", "id": 4, "can_network": False},
                4: {"name": "Step 2-文学大师版", "id": 5, "can_network": False}
            }
            self.current_model_id = self.config.get('current_model_id', 6)
            
            # 初始化图片识别相关配置
            self.image_config = self.config.get('image_config', {})
            self.imgprompt = self.image_config.get('imgprompt', '解释下图片内容')
            self.pic_trigger_prefix = self.image_config.get('trigger', '识图')  # 获取识图触发词
            
            # 初始化图片识别状态管理
            self.waiting_for_image = {}  # 等待上传图片的状态
            self.image_prompts = {}  # 图片提示词
            self.multi_image_data = {}  # 多图片数据存储 {waiting_id: {'count': 需要的图片数量, 'current': 当前收到的数量, 'images': [], 'prompt': ''}}
            self.max_images = 3  # 最大支持的图片数量
            
            # 添加参考图视频功能状态管理
            self.video_ref_waiting = {}  # 等待参考图上传的状态 {waiting_id: {'prompt': 提示词, 'use_rephrase': 是否润色, 'camera_list': 镜头语言列表}}
            
            # 定义镜头语言映射
            self.camera_movements = {
                "拉近": "镜头拉近",
                "拉远": "镜头拉远",
                "向左": "镜头向左",
                "向右": "镜头向右",
                "向上": "镜头向上",
                "向下": "镜头向下",
                "禁止": "镜头禁止"
            }
            
            # 添加Token刷新时间记录
            self.last_token_refresh = 0
            
            if self.client is None:
                self.client = httpx.Client(http2=True, timeout=30.0)
                
            # 尝试同步服务器状态
            self._sync_server_state()
            
            # 添加最近消息记录
            self.last_message = {
                'chat_id': None,
                'messages': [],
                'last_time': 0
            }
            
            logger.info("[Yuewen] inited")
        except Exception as e:
            logger.error(f"[Yuewen] 初始化失败: {str(e)}")
            raise e

    def on_receive_message(self, e_context: EventContext):
        """处理接收到的消息"""
        context = e_context['context']
        
        # 只处理图片类型消息
        if context.type != ContextType.IMAGE:
            return
        
        # 获取用户信息
        msg = context.kwargs.get("msg")
        is_group = context.kwargs.get("isgroup", False)
        
        # 生成等待ID
        if is_group:
            group_id = msg.other_user_id if msg else None
            real_user_id = msg.actual_user_id if msg and hasattr(msg, "actual_user_id") else None
            waiting_id = f"{group_id}_{real_user_id}" if real_user_id else group_id
        else:
            real_user_id = msg.from_user_id if msg else None
            waiting_id = real_user_id
        
        # 处理参考图生成视频的图片上传
        if waiting_id in self.video_ref_waiting:
            logger.debug("[Yuewen] 收到参考图片，准备生成视频")
            if hasattr(context, 'kwargs') and 'msg' in context.kwargs:
                msg = context.kwargs['msg']
                if hasattr(msg, '_prepare_fn') and not msg._prepared:
                    try:
                        msg._prepare_fn()
                        msg._prepared = True
                        if hasattr(msg, 'content'):
                            logger.debug(f"[Yuewen] 参考图片准备完成，保存路径: {msg.content}")
                            # 获取参考图配置
                            video_ref_config = self.video_ref_waiting.pop(waiting_id)
                            # 处理参考图生成视频
                            result = self._handle_video_ref_image(
                                msg.content,
                                video_ref_config['prompt'],
                                e_context,
                                video_ref_config['use_rephrase'],
                                video_ref_config.get('camera_list', [])
                            )
                            # 只有在处理失败且返回错误消息时才回复
                            if result:
                                reply = Reply()
                                reply.type = ReplyType.TEXT
                                reply.content = result
                                e_context["channel"].send(reply, e_context["context"])
                            e_context.action = EventAction.BREAK_PASS
                            return
                    except Exception as e:
                        logger.error(f"[Yuewen] 参考图片处理失败: {e}")
                        # 清理状态
                        self.video_ref_waiting.pop(waiting_id, None)
                        # 发送错误消息
                        reply = Reply()
                        reply.type = ReplyType.TEXT
                        reply.content = "❌ 参考图片处理失败，请重试"
                        e_context["channel"].send(reply, e_context["context"])
                        e_context.action = EventAction.BREAK_PASS
                        return
        
        # 只有当用户在等待上传图片状态时才处理
        if waiting_id in self.waiting_for_image or waiting_id in self.multi_image_data:
            logger.debug("[Yuewen] 收到等待中的图片消息")
            if hasattr(context, 'kwargs') and 'msg' in context.kwargs:
                msg = context.kwargs['msg']
                if hasattr(msg, '_prepare_fn') and not msg._prepared:
                    try:
                        msg._prepare_fn()
                        msg._prepared = True
                        if hasattr(msg, 'content'):
                            logger.debug(f"[Yuewen] 图片准备完成，保存路径: {msg.content}")
                    except Exception as e:
                        logger.error(f"[Yuewen] 图片准备失败: {e}")
        else:
            # 关键修改：明确忽略不需要处理的图片
            logger.info("[Yuewen] 不在等待图片状态，忽略图片消息")
            e_context.action = EventAction.CONTINUE

    def _generate_traceparent(self):
        trace_id = ''.join(random.choices('0123456789abcdef', k=32))
        span_id = ''.join(random.choices('0123456789abcdef', k=16))
        return f"00-{trace_id}-{span_id}-01"

    def _generate_tracestate(self):
        return f"yuewen@rsid={random.getrandbits(64):016x}"

    def _update_headers(self):
        headers = self.login_handler.base_headers.copy()
        headers.update({
            'Cookie': f"Oasis-Webid={self.config['oasis_webid']}; Oasis-Token={self.config['oasis_token']}",
            'oasis-webid': self.config['oasis_webid'],
            'x-rum-traceparent': self._generate_traceparent(),
            'x-rum-tracestate': self._generate_tracestate()
        })
        return headers

    def _handle_commands(self, message):
        msg = message.strip().lower()
      
        if msg.startswith("切换模型"):
            try:
                model_num = int(msg.split()[-1])
                return self._switch_model(model_num)
            except:
                return "⚠️ 无效的模型编号，使用「打印模型」查看"
      
        if msg == "打印模型":
            return self._list_models()
      
        current_model = next((m for m in self.models.values() if m['id'] == self.current_model_id), None)
        if msg in ["开启联网", "关闭联网"]:
            if not current_model or not current_model['can_network']:
                return "⚠️ 当前模型不支持联网"
            return self._handle_network_command(msg)
      
        return None

    def _switch_model(self, model_num):
        if model_num not in self.models:
            return f"⚠️ 无效模型编号，可用：{', '.join(map(str, self.models.keys()))}"
        
        model_info = self.models[model_num]
        if self.current_model_id == model_info['id']:
            return f"✅ 已经是{model_info['name']}模型"
        
        # 先尝试刷新令牌
        if not self.login_handler.refresh_token():
            return "⚠️ 令牌刷新失败，请重试"
        
        # 切换模型
        if not self._call_set_model(model_info['id']):
            return "⚠️ 模型切换失败，请重试"
        
        self.current_model_id = model_info['id']
        self.config['current_model_id'] = self.current_model_id
        self.login_handler.save_config()
        
        # 如果是deepseek r1模型，强制开启联网模式
        if model_info['id'] == 6:
            self.config['network_mode'] = True
            if not self._enable_search(True):
                logger.warning("[Yuewen] 联网模式启用失败")
            if not self._enable_deep_thinking():
                logger.warning("[Yuewen] 深度思考模式启用失败")
        
        # 创建新会话
        self.current_chat_id = None
        if not self.create_chat():
            return "⚠️ 新会话创建失败"
        
        # 最后再次同步确认服务器状态
        if not self._sync_server_state():
            logger.warning("[Yuewen] 服务器状态同步失败")
        
        return f"✅ 已切换至{model_info['name']}"

    def _call_set_model(self, model_id):
        """切换模型"""
        for retry in range(2):
            headers = self._update_headers()
            headers['Content-Type'] = 'application/json'
            try:
                response = self.client.post(
                    'https://yuewen.cn/api/proto.user.v1.UserService/SetModelInUse',
                    headers=headers,
                    json={"modelId": model_id}
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("result") == "RESULT_CODE_SUCCESS":
                        return True
                elif response.status_code == 401 and retry == 0:
                    if self.login_handler.refresh_token():
                        continue
                    else:
                        logger.error("[Yuewen] Token刷新失败")
                logger.error(f"[Yuewen] 模型切换失败: {response.text}")
                return False
            except Exception as e:
                if retry == 0:
                    continue
                logger.error(f"[Yuewen] 模型切换失败: {str(e)}")
                return False
        return False

    def _list_models(self):
        output = ["可用模型："]
        for num, info in self.models.items():
            status = "（支持联网）" if info['can_network'] else ""
            current = " ← 当前使用" if info['id'] == self.current_model_id else ""
            output.append(f"{num}. {info['name']}{status}{current}")
        return '\n'.join(output)

    def _enable_search(self, enable: bool):
        """切换联网模式"""
        for retry in range(2):
            headers = self._update_headers()
            headers.update({
                'Content-Type': 'application/json',
                'referer': f'https://yuewen.cn/chats/{self.current_chat_id}' if self.current_chat_id else 'https://yuewen.cn/chats/new',
                'canary': 'false',
                'connect-protocol-version': '1',
                'oasis-appid': '10200',
                'oasis-mode': '2',
                'oasis-platform': 'web',
                'priority': 'u=1, i',
                'x-waf-client-type': 'fetch_sdk'
            })
            endpoint = "EnableSearch" if enable else "DisableSearch"
            try:
                response = self.client.post(
                    f'https://yuewen.cn/api/proto.user.v1.UserService/{endpoint}',
                    headers=headers,
                    json={}
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("result") == "RESULT_CODE_SUCCESS":
                        logger.info(f"[Yuewen] 联网模式{'启用' if enable else '关闭'}成功")
                        return True
                    elif not data:  # 空响应也视为成功
                        logger.info(f"[Yuewen] 联网模式{'启用' if enable else '关闭'}成功(空响应)")
                        return True
                elif response.status_code == 401 and retry == 0:
                    if self.login_handler.refresh_token():
                        continue
                    else:
                        logger.error("[Yuewen] Token刷新失败")
                logger.error(f"[Yuewen] 联网模式切换失败: {response.text}")
                return False
            except Exception as e:
                if retry == 0:
                    continue
                logger.error(f"[Yuewen] 联网模式切换失败: {str(e)}")
                return False
        return False

    def _handle_network_command(self, message):
        enable = message == "开启联网"
        
        # 先尝试刷新令牌
        if not self.login_handler.refresh_token():
            return "⚠️ 令牌刷新失败，请重试"
            
        # 如果当前没有会话，先创建会话
        if not self.current_chat_id:
            if not self.create_chat():
                return "⚠️ 会话创建失败"
            
        if self._enable_search(enable):
            self.config['network_mode'] = enable
            self.login_handler.save_config()
            status = "启用" if enable else "关闭"
            return f"✅ 联网模式已{status}"
        return f"⚠️ 联网模式{'启用' if enable else '关闭'}失败"

    def create_chat(self, chat_name="新话题"):
        """创建新会话"""
        for retry in range(2):
            headers = self._update_headers()
            headers['Content-Type'] = 'application/json'
            try:
                response = self.client.post(
                    'https://yuewen.cn/api/proto.chat.v1.ChatService/CreateChat',
                    headers=headers,
                    json={'chatName': chat_name}
                )
                if response.status_code == 200:
                    data = response.json()
                    if 'chatId' in data:
                        self.current_chat_id = data['chatId']
                        self.last_active_time = time.time()
                        logger.info(f"[Yuewen] 新建会话成功 ID: {self.current_chat_id}")
                        return True
                elif response.status_code == 401 and retry == 0:
                    if self.login_handler.refresh_token():
                        continue
                    else:
                        logger.error("[Yuewen] Token刷新失败")
                logger.error(f"[Yuewen] 创建会话失败: {response.text}")
                return False
            except Exception as e:
                if retry == 0:
                    continue
                logger.error(f"[Yuewen] 创建会话失败: {str(e)}")
                return False
        return False

    def _construct_protocol_packet(self, message):
        payload = {
            "chatId": self.current_chat_id,
            "messageInfo": {
                "text": message,
                "author": {"role": "user"}
            },
            "messageMode": "SEND_MESSAGE",
            "modelId": self.current_model_id
        }
        json_str = json.dumps(payload, separators=(',', ':'), ensure_ascii=False)
        encoded = json_str.encode('utf-8')
        protocol_header = struct.pack('>BI', 0, len(encoded))
        return protocol_header + encoded

    def _parse_stream_response(self, response, start_time):
        buffer = bytearray()
        text_buffer = []
        has_thinking_stage = False  # 是否包含思考阶段
        is_done = False  # 是否完成
        user_message_id = None  # 记录用户消息ID
        ai_message_id = None  # 记录AI回答消息ID
        
        try:
            # 获取当前模型信息
            current_model = next((m for m in self.models.values() if m['id'] == self.current_model_id), None)
            model_name = current_model['name'] if current_model else f"未知模型(ID: {self.current_model_id})"
            
            logger.debug(f"[Yuewen] 开始处理响应，使用模型: {model_name}")
            logger.debug(f"[Yuewen] 当前会话ID: {self.current_chat_id}")
            
            for chunk in response.iter_bytes():
                buffer.extend(chunk)
                while len(buffer) >= 5:
                    try:
                        msg_type, length = struct.unpack('>BI', buffer[:5])
                    except struct.error:
                        buffer.clear()
                        break

                    if len(buffer) < 5 + length:
                        break

                    packet = buffer[5:5+length]
                    buffer = buffer[5+length:]

                    try:
                        data = json.loads(packet.decode('utf-8'))
                        
                        # 检查是否包含思考阶段
                        if 'textEvent' in data:
                            event = data['textEvent']
                            if event.get('stage') == 'TEXT_STAGE_THINKING':
                                has_thinking_stage = True
                                continue
                            
                            # 如果有stage字段且不是SOLUTION阶段，跳过
                            if event.get('stage') and event.get('stage') != 'TEXT_STAGE_SOLUTION':
                                continue
                                
                            content = event.get('text', '')
                            if content:
                                text_buffer.append(content)
                                
                        # 记录消息ID - 从startEvent中获取
                        if 'startEvent' in data:
                            start_event = data['startEvent']
                            ai_message_id = start_event.get('messageId')
                            parent_id = start_event.get('parentMessageId')
                            if parent_id:
                                user_message_id = parent_id
                        
                        # 检查是否完成
                        if 'doneEvent' in data:
                            is_done = True
                            
                    except Exception as e:
                        logger.error(f"[Yuewen] 解析数据包失败: {e}")
                        continue

            # 如果响应未完成，返回错误
            if not is_done:
                return "响应未完成，请重试"

            cost_time = time.time() - start_time
            # 优化换行格式处理
            final_text = ''.join(text_buffer)
            
            # 处理特殊字符和格式
            final_text = (
                final_text.replace('\u200b', '')      # 移除零宽空格
                .replace('\r\n', '\n')                # 统一换行符
                .replace('\r', '\n')                  # 处理旧版Mac换行
            )
            
            # 处理markdown格式的列表
            final_text = re.sub(r'\n(\d+\.|\-|\*)\s*', r'\n\n\1 ', final_text)
            
            # 处理连续换行，但保留markdown格式
            lines = final_text.split('\n')
            processed_lines = []
            for i, line in enumerate(lines):
                if i > 0 and (line.startswith('- ') or line.startswith('* ') or re.match(r'^\d+\.\s', line)):
                    processed_lines.append('')  # 在列表项前添加空行
                processed_lines.append(line)
            final_text = '\n'.join(processed_lines)
            
            # 清理多余的连续换行
            while '\n\n\n' in final_text:
                final_text = final_text.replace('\n\n\n', '\n\n')
            
            # 保留段落格式但去除首尾空白
            final_text = final_text.strip()
            
            # 更新最近消息记录
            if self.current_chat_id and user_message_id and ai_message_id:
                logger.debug(f"[Yuewen] 记录消息ID - User: {user_message_id}, AI: {ai_message_id}")
                self.last_message = {
                    'chat_id': self.current_chat_id,
                    'messages': [
                        {'messageId': ai_message_id, 'messageIndex': 1},
                        {'messageId': user_message_id}
                    ],
                    'last_time': time.time()
                }
          
            if final_text:
                # 获取联网状态
                network_mode = "联网" if self.config.get('network_mode', False) else "未联网"
                # 构建状态信息
                status_info = f"使用{model_name}模型{network_mode}模式回答（耗时{cost_time:.2f}秒）：\n"
                return f"{status_info}{final_text}\n\n3分钟内发送yw分享获取回答图片"
            return f"未收到有效回复（耗时{cost_time:.2f}秒）"
        except Exception as e:
            logger.error(f"[Yuewen] 解析响应失败: {e}")
            return f"响应解析失败: {str(e)}"

    def send_message(self, message):
        """消息发送主逻辑"""
        cmd_response = self._handle_commands(message)
        if cmd_response:
            return cmd_response

        if not self.current_chat_id or (time.time() - self.last_active_time) > 180:
            if not self.create_chat():
                return "会话创建失败"

        for retry in range(2):
            try:
                protocol_data = self._construct_protocol_packet(message)
                start_time = time.time()
                headers = self._update_headers()
                headers['Content-Type'] = 'application/connect+json'
                
                response = self.client.post(
                    'https://yuewen.cn/api/proto.chat.v1.ChatMessageService/SendMessageStream',
                    headers=headers,
                    content=protocol_data,
                    timeout=30
                )

                if response.status_code != 200:
                    if response.status_code == 401 and retry == 0:
                        if self.login_handler.refresh_token():
                            continue
                        else:
                            logger.error("[Yuewen] Token刷新失败")
                    return self._handle_error(response)

                self.last_active_time = time.time()
                return self._parse_stream_response(response, start_time)

            except (httpx.HTTPError, struct.error) as e:
                if retry == 0 and self.login_handler.refresh_token():
                    continue
                return "请求失败，请重试"
            except Exception as e:
                return f"处理错误: {str(e)}"

    def _handle_error(self, response):
        try:
            error = response.json()
            return f"服务错误: {error.get('error', '未知错误')}"
        except:
            return f"HTTP错误 {response.status_code}: {response.text[:200]}"

    def _get_image_data(self, msg, content):
        """获取图片数据"""
        try:
            # 如果已经是二进制数据，直接返回
            if isinstance(content, bytes):
                logger.debug(f"[Yuewen] 处理二进制数据，大小: {len(content)} 字节")
                return content

            logger.debug(f"[Yuewen] 开始处理图片，类型: {type(content)}")
            
            # 统一的文件读取函数
            def read_file(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        data = f.read()
                        logger.debug(f"[Yuewen] 成功读取文件: {file_path}, 大小: {len(data)} 字节")
                        return data
                except Exception as e:
                    logger.error(f"[Yuewen] 读取文件失败 {file_path}: {e}")
                    return None
            
            # 按优先级尝试不同的读取方式
            if isinstance(content, str):
                # 1. 如果是文件路径，直接读取
                if os.path.isfile(content):
                    data = read_file(content)
                    if data:
                        return data
                
                # 2. 如果是URL，尝试下载
                if content.startswith(('http://', 'https://')):
                    logger.debug(f"[Yuewen] 尝试从URL下载: {content}")
                    try:
                        response = requests.get(content, timeout=30)
                        if response.status_code == 200:
                            return response.content
                    except Exception as e:
                        logger.error(f"[Yuewen] 从URL下载失败: {e}")
            
            # 3. 尝试从msg.content读取
            if hasattr(msg, 'content') and os.path.isfile(msg.content):
                data = read_file(msg.content)
                if data:
                    return data
            
            # 4. 如果文件未下载，尝试下载
            if hasattr(msg, '_prepare_fn') and not msg._prepared:
                logger.debug("[Yuewen] 尝试下载图片...")
                try:
                    msg._prepare_fn()
                    msg._prepared = True
                    time.sleep(1)  # 等待文件准备完成
                    
                    if hasattr(msg, 'content') and os.path.isfile(msg.content):
                        data = read_file(msg.content)
                        if data:
                            return data
                except Exception as e:
                    logger.error(f"[Yuewen] 下载图片失败: {e}")
            
            logger.error(f"[Yuewen] 无法获取图片数据")
            return None
            
        except Exception as e:
            logger.error(f"[Yuewen] 获取图片数据失败: {e}")
            return None

    def _process_image(self, image_path, prompt, e_context):
        """处理图片识别请求"""
        try:
            start_time = time.time()  # 在开始处理时就开始计时
            # 获取图片数据
            msg = e_context['context'].kwargs.get('msg')
            logger.info("[Yuewen] 开始处理图片识别请求")
            
            image_bytes = self._get_image_data(msg, image_path)
            if not image_bytes:
                return "获取图片失败，请重试"

            # 上传图片
            file_id = self._upload_image(image_bytes)
            if not file_id:
                return "图片上传失败，请重试"
            
            # 如果没有当前会话或会话已超时，才创建新会话
            if not self.current_chat_id or (time.time() - self.last_active_time) > 180:
                if not self.create_chat():
                    return "创建会话失败，请重试"
            
            # 获取图片尺寸
            try:
                import io
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes))
                width, height = img.size
                size = len(image_bytes)
                logger.info(f"[Yuewen] 图片尺寸: {width}x{height}, 大小: {size} 字节")
            except:
                width = height = 800
                size = len(image_bytes)
                logger.warning("[Yuewen] 无法获取图片尺寸，使用默认值")
            
            # 构建带图片的消息
            message = {
                "chatId": self.current_chat_id,
                "messageInfo": {
                    "text": prompt,
                    "attachments": [{
                        "attachmentType": "image/jpeg",
                        "attachmentId": file_id,
                        "name": f"n_v{random.getrandbits(128):032x}.jpg",
                        "width": str(width),
                        "height": str(height),
                        "size": str(size),
                        "usedPercent": -1
                    }],
                    "author": {"role": "user"}
                },
                "messageMode": "SEND_MESSAGE",
                "modelId": self.current_model_id
            }
            
            # 发送消息
            json_str = json.dumps(message, separators=(',', ':'), ensure_ascii=False)
            encoded = json_str.encode('utf-8')
            protocol_header = struct.pack('>BI', 0, len(encoded))
            
            headers = self._update_headers()
            headers['Content-Type'] = 'application/connect+json'
            
            response = self.client.post(
                'https://yuewen.cn/api/proto.chat.v1.ChatMessageService/SendMessageStream',
                headers=headers,
                content=protocol_header + encoded
            )
            
            if response.status_code != 200:
                logger.error(f"[Yuewen] 发送图片消息失败: HTTP {response.status_code}")
                return "请求失败，请重试"
            
            self.last_active_time = time.time()  # 更新最后活动时间
            return self._parse_stream_response(response, start_time)
            
        except Exception as e:
            logger.error(f"[Yuewen] 处理图片失败: {e}")
            return "处理失败，请重试"

    def _upload_image(self, image_bytes):
        """上传图片到悦问服务器"""
        try:
            if not image_bytes:
                logger.error("[Yuewen] 图片数据为空")
                return None
            
            file_size = len(image_bytes)
            logger.debug(f"[Yuewen] 准备上传图片，大小: {file_size} 字节")
            
            # 生成随机文件名
            file_name = f"n_v{random.getrandbits(128):032x}.jpg"
            logger.debug(f"[Yuewen] 生成的文件名: {file_name}")
            
            # 准备上传请求
            headers = self._update_headers()
            headers.update({
                'accept': '*/*',
                'accept-language': 'zh-CN,zh;q=0.9',
                'cache-control': 'no-cache',
                'content-type': 'image/jpeg',
                'content-length': str(file_size),
                'oasis-appid': '10200',
                'oasis-platform': 'web',
                'origin': 'https://yuewen.cn',
                'pragma': 'no-cache',
                'priority': 'u=1, i',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
                'stepchat-meta-size': str(file_size),
                'x-waf-client-type': 'fetch_sdk'
            })
            
            if self.current_chat_id:
                headers['referer'] = f'https://yuewen.cn/chats/{self.current_chat_id}'
            
            # 上传图片，带令牌检查和重试逻辑
            for retry in range(2):
                try:
                    # 发送上传请求
                    upload_url = f'https://yuewen.cn/api/storage?file_name={file_name}'
                    logger.debug(f"[Yuewen] 开始上传图片到: {upload_url}")
                    
                    response = self.client.put(upload_url, headers=headers, content=image_bytes)
                    
                    if response.status_code == 200:
                        file_id = response.json().get('id')
                        if file_id:
                            logger.debug(f"[Yuewen] 文件上传成功，ID: {file_id}")
                            return file_id
                    elif response.status_code == 401 and retry == 0:
                        # 尝试刷新token
                        if self.login_handler.refresh_token():
                            # 更新headers中的token
                            headers = self._update_headers()
                            headers.update({
                                'accept': '*/*',
                                'accept-language': 'zh-CN,zh;q=0.9',
                                'cache-control': 'no-cache',
                                'content-type': 'image/jpeg',
                                'content-length': str(file_size),
                                'oasis-appid': '10200',
                                'oasis-platform': 'web',
                                'origin': 'https://yuewen.cn',
                                'pragma': 'no-cache',
                                'priority': 'u=1, i',
                                'sec-fetch-dest': 'empty',
                                'sec-fetch-mode': 'cors',
                                'sec-fetch-site': 'same-origin',
                                'stepchat-meta-size': str(file_size),
                                'x-waf-client-type': 'fetch_sdk'
                            })
                            if self.current_chat_id:
                                headers['referer'] = f'https://yuewen.cn/chats/{self.current_chat_id}'
                            continue
                        else:
                            logger.error("[Yuewen] Token刷新失败")
                    else:
                        logger.error(f"[Yuewen] 上传失败: HTTP {response.status_code}")
                        
                except Exception as e:
                    if retry == 0:
                        logger.warning(f"[Yuewen] 上传出错，准备重试: {e}")
                        continue
                    logger.error(f"[Yuewen] 上传失败: {e}")
                    break
                    
            return None
            
        except Exception as e:
            logger.error(f"[Yuewen] 上传图片失败: {e}")
            return None

    def on_handle_context(self, e_context: EventContext):
        """处理上下文"""
        # 图片消息只由 on_receive_message 处理，不在这里处理
        if e_context['context'].type == ContextType.IMAGE:
            # 如果不是我们正在等待的特定图片，直接跳过
            msg = e_context['context'].kwargs.get("msg")
            is_group = e_context['context'].kwargs.get("isgroup", False)
            
            # 生成等待ID
            if is_group:
                group_id = msg.other_user_id if msg else None
                real_user_id = msg.actual_user_id if msg and hasattr(msg, "actual_user_id") else None
                waiting_id = f"{group_id}_{real_user_id}" if real_user_id else group_id
            else:
                real_user_id = msg.from_user_id if msg else None
                waiting_id = real_user_id
                
            # 如果不在等待状态，直接返回
            if waiting_id not in self.waiting_for_image and waiting_id not in self.multi_image_data and waiting_id not in self.video_ref_waiting:
                e_context.action = EventAction.CONTINUE
                return
        
        # 只处理文本或特定等待中的图片消息
        if e_context['context'].type != ContextType.TEXT and e_context['context'].type != ContextType.IMAGE:
            return

        content = e_context['context'].content
        if not content:
            return

        # 只有在处理文本命令时才同步状态，防止对每个图片都同步
        if e_context['context'].type == ContextType.TEXT:
            # 在处理消息前同步状态
            if time.time() - self.last_active_time > 180:  # 3分钟没活动就检查同步
                self._sync_server_state()
        
        # 获取用户信息
        msg = e_context['context'].kwargs.get("msg")
        is_group = e_context['context'].kwargs.get("isgroup", False)
        
        # 生成等待ID
        if is_group:
            group_id = msg.other_user_id if msg else None
            real_user_id = msg.actual_user_id if msg and hasattr(msg, "actual_user_id") else None
            waiting_id = f"{group_id}_{real_user_id}" if real_user_id else group_id
        else:
            real_user_id = msg.from_user_id if msg else None
            waiting_id = real_user_id

        # 处理图片消息
        if e_context['context'].type == ContextType.IMAGE and waiting_id in self.multi_image_data:
            try:
                multi_data = self.multi_image_data[waiting_id]
                # 获取图片数据
                image_data = self._get_image_data(msg, content)
                if not image_data:
                    reply = Reply()
                    reply.type = ReplyType.TEXT
                    reply.content = "获取图片失败，请重试"
                    e_context['reply'] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 上传图片
                file_id = self._upload_image(image_data)
                if not file_id:
                    reply = Reply()
                    reply.type = ReplyType.TEXT
                    reply.content = "图片上传失败，请重试"
                    e_context['reply'] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 检查文件状态
                if not self._check_file_status(file_id):
                    reply = Reply()
                    reply.type = ReplyType.TEXT
                    reply.content = "图片处理失败，请重试"
                    e_context['reply'] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                # 获取图片尺寸
                try:
                    import io
                    from PIL import Image
                    img = Image.open(io.BytesIO(image_data))
                    width, height = img.size
                    size = len(image_data)
                except:
                    width = height = 800
                    size = len(image_data)
                
                # 添加图片信息
                multi_data['images'].append({
                    'file_id': file_id,
                    'width': width,
                    'height': height,
                    'size': size
                })
                multi_data['current'] += 1
                
                # 如果图片数量达到要求
                if multi_data['current'] >= multi_data['count']:
                    # 处理所有图片
                    result = self._process_multi_images(multi_data['images'], multi_data['prompt'], e_context)
                    reply = Reply()
                    reply.type = ReplyType.TEXT
                    reply.content = result
                    e_context['reply'] = reply
                    e_context.action = EventAction.BREAK_PASS
                    # 清理状态
                    del self.multi_image_data[waiting_id]
                else:
                    # 继续等待下一张图片
                    reply = Reply()
                    reply.type = ReplyType.TEXT
                    reply.content = f"请发送第{multi_data['current'] + 1}张图片"
                    e_context['reply'] = reply
                    e_context.action = EventAction.BREAK_PASS
                return
                
            except Exception as e:
                logger.error(f"[Yuewen] 处理图片失败: {e}")
                reply = Reply()
                reply.type = ReplyType.TEXT
                reply.content = "处理图片失败，请重试"
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS
                # 清理状态
                self.multi_image_data.pop(waiting_id, None)
                return

        trigger = self.config.get("trigger_prefix", "yw")
        # 增加容错处理,移除多余空格并统一格式
        content = content.strip()
        if not content.lower().startswith(trigger.lower()):
            return
            
        # 移除触发词,处理多余空格
        content = re.sub(f'^{trigger}\\s*', '', content, flags=re.IGNORECASE).strip()
        
        # 获取用户ID
        msg = e_context['context'].kwargs.get("msg")
        is_group = e_context['context'].kwargs.get("isgroup", False)
        
        # 生成用户ID
        if is_group:
            group_id = msg.other_user_id if msg else None
            real_user_id = msg.actual_user_id if msg and hasattr(msg, "actual_user_id") else None
            user_id = f"{group_id}_{real_user_id}" if real_user_id else group_id
        else:
            real_user_id = msg.from_user_id if msg else None
            user_id = real_user_id
        
        # 检查是否是登录命令
        if content.lower() == "登录":
            return self._initiate_login(e_context, user_id)
        
        # 检查是否正在等待验证码
        if user_id in self.waiting_for_verification:
            if content.isdigit() and len(content) == 4:
                return self._verify_login(e_context, user_id, content)
            elif len(content) == 11 and content.isdigit():
                return self._send_verification_code(e_context, user_id, content)
        
        # 检查登录状态，如果配置文件丢失或token无效，自动提示登录
        if not self._check_login_status():
            # 如果是可自动注册的场景，尝试自动注册
            if not self.config.get('oasis_webid'):
                if self.login_handler.register_device():
                    logger.info("[Yuewen] 设备自动注册成功")
                else:
                    logger.error("[Yuewen] 设备自动注册失败")
                    
            if not self.is_login_triggered:
                self.is_login_triggered = True
                reply = Reply()
                reply.type = ReplyType.TEXT
                reply.content = "⚠️ 跃问账号未登录或已失效，请先发送\"yw登录\"进行登录"
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS
                return
        
        if not content:
            reply = Reply()
            reply.type = ReplyType.TEXT
            reply.content = "请输入要发送给跃问的消息"
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS
            return

        # 处理参考图生成视频命令
        if content.lower().startswith('参考图'):
            result = self._handle_video_ref_request(content, e_context, user_id)
            reply = Reply()
            reply.type = ReplyType.TEXT
            reply.content = result
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS
            return

        # 处理识图命令,增加容错处理
        pic_trigger = self.pic_trigger_prefix
        
        # 匹配识图命令模式
        img_cmd_match = re.match(f'^{pic_trigger}\\s*(\\d)?\\s*(.*)$', content, re.IGNORECASE)
        if img_cmd_match:
            img_count = int(img_cmd_match.group(1)) if img_cmd_match.group(1) else 1
            prompt = img_cmd_match.group(2).strip() if img_cmd_match.group(2) else self.imgprompt
            
            # 检查图片数量是否有效
            if img_count < 1 or img_count > self.max_images:
                reply = Reply()
                reply.type = ReplyType.TEXT
                reply.content = f"图片数量必须在1-{self.max_images}之间"
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS
                return
            
            # 检查是否有引用图片
            ref_type = e_context["context"].get("ref_type")
            ref_content = e_context["context"].get("ref_content")
            
            if ref_type == "image" and ref_content:
                logger.info("[Yuewen] 检测到引用图片，直接处理")
                try:
                    result = self._process_image(ref_content, prompt, e_context)
                    reply = Reply()
                    reply.type = ReplyType.TEXT
                    reply.content = result
                    e_context['reply'] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
                except Exception as e:
                    logger.error(f"[Yuewen] 处理引用图片失败: {e}")
                    reply = Reply()
                    reply.type = ReplyType.TEXT
                    reply.content = "处理引用图片失败，请重试"
                    e_context['reply'] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
            
            # 初始化多图片上传状态
            self.multi_image_data[waiting_id] = {
                'count': img_count,
                'current': 0,
                'images': [],
                'prompt': prompt
            }
            
            reply = Reply()
            reply.type = ReplyType.TEXT
            reply.content = f"请发送第1张图片" if img_count > 1 else "请发送图片"
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS
            return

        # 处理其他命令,增加容错处理
        if content.lower() == "新建会话":
            self.current_chat_id = None
            reply = Reply()
            reply.type = ReplyType.TEXT
            if self.create_chat():
                reply.content = "✅ 新会话已创建"
            else:
                reply.content = "❌ 新会话创建失败"
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS
            return

        # 处理分享命令
        if content.lower() == "分享":
            if (time.time() - self.last_message['last_time']) > 180:  # 超过3分钟
                reply = Reply()
                reply.type = ReplyType.TEXT
                reply.content = "请先发起对话吧"
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS
                return
                
            if not self.last_message['chat_id'] or not self.last_message['messages']:
                reply = Reply()
                reply.type = ReplyType.TEXT
                reply.content = "没有可分享的对话"
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS
                return
                
            image_url = self._get_share_image(
                self.last_message['chat_id'],
                self.last_message['messages']
            )
            
            if not image_url:
                reply = Reply()
                reply.type = ReplyType.TEXT
                reply.content = "获取分享图片失败"
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS
                return
                
            reply = Reply()
            reply.type = ReplyType.IMAGE_URL
            reply.content = image_url
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS
            return

        # 处理文生视频命令
        if content.lower().startswith('视频'):
            video_prompt = content[2:].strip()  # 去掉"视频"两个字
            result = self._handle_video_request(video_prompt, e_context)
            if result:  # 只有在处理失败时才需要回复
                reply = Reply()
                reply.type = ReplyType.TEXT
                reply.content = result
                e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS
            return

        try:
            response = self.send_message(content)
            reply = Reply()
            reply.type = ReplyType.TEXT
            reply.content = response
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS
        except Exception as e:
            reply = Reply()
            reply.type = ReplyType.TEXT
            reply.content = f"处理消息时发生错误: {str(e)}"
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS

    def _enable_deep_thinking(self):
        """开启深度思考模式"""
        for retry in range(2):
            headers = self._update_headers()
            headers['Content-Type'] = 'application/json'
            headers['referer'] = f'https://yuewen.cn/chats/{self.current_chat_id}' if self.current_chat_id else 'https://yuewen.cn/chats/new'
            try:
                response = self.client.post(
                    'https://yuewen.cn/api/proto.user.v1.UserService/EnableDeepThinking',
                    headers=headers,
                    json={}
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("modelInUse", {}).get("id") == 6:
                        logger.info("[Yuewen] 深度思考模式启用成功")
                        return True
                elif response.status_code == 401 and retry == 0:
                    if self.login_handler.refresh_token():
                        continue
                    else:
                        logger.error("[Yuewen] Token刷新失败")
                logger.error(f"[Yuewen] 深度思考模式启用失败: {response.text}")
                return False
            except Exception as e:
                if retry == 0:
                    continue
                logger.error(f"[Yuewen] 深度思考模式启用失败: {str(e)}")
                return False
        return False

    def _sync_server_state(self):
        """同步服务器端的模型和联网状态"""
        for retry in range(2):  # 添加重试机制
            try:
                logger.info("[Yuewen] 开始同步服务器状态")
                headers = self._update_headers()
                headers['Content-Type'] = 'application/json'
                
                # 获取服务器端用户状态
                response = self.client.post(
                    'https://yuewen.cn/api/proto.user.v1.UserService/GetUser',
                    headers=headers,
                    json={}
                )
                
                if response.status_code == 401 and retry == 0:
                    # 尝试刷新令牌
                    if self.login_handler.refresh_token():
                        logger.info("[Yuewen] 令牌刷新成功,重试获取状态")
                        continue
                    else:
                        logger.error("[Yuewen] Token刷新失败")
                        return False
                
                if response.status_code != 200:
                    logger.error(f"[Yuewen] 获取用户状态失败: HTTP {response.status_code}")
                    return False
                    
                data = response.json()
                if 'user' not in data:
                    logger.error("[Yuewen] 获取用户状态失败: 无效响应")
                    return False
                    
                user_data = data['user']
                server_model_id = user_data.get('modelInUse')
                server_search_enabled = user_data.get('enableSearch', False)
                
                changes_made = False
                
                # 检查模型是否需要同步
                if server_model_id != self.current_model_id:
                    logger.info(f"[Yuewen] 检测到模型不一致: 本地({self.current_model_id}) vs 服务器({server_model_id})")
                    if self._call_set_model(self.current_model_id):
                        logger.info(f"[Yuewen] 已将服务器模型同步为: {self.current_model_id}")
                        changes_made = True
                        
                        # 如果是 r1 模型，确保启用深度思考
                        if self.current_model_id == 6:
                            if self._enable_deep_thinking():
                                logger.info("[Yuewen] 已启用深度思考模式")
                            else:
                                logger.warning("[Yuewen] 深度思考模式启用失败")
                    else:
                        logger.error("[Yuewen] 模型同步失败")
                        return False
                
                # 检查联网状态是否需要同步
                current_model = next((m for m in self.models.values() if m['id'] == self.current_model_id), None)
                if current_model and current_model['can_network']:
                    config_network_mode = self.config.get('network_mode', True)
                    if server_search_enabled != config_network_mode:
                        logger.info(f"[Yuewen] 检测到联网状态不一致: 本地({config_network_mode}) vs 服务器({server_search_enabled})")
                        if self._enable_search(config_network_mode):
                            logger.info(f"[Yuewen] 已将服务器联网状态同步为: {config_network_mode}")
                            changes_made = True
                        else:
                            logger.error("[Yuewen] 联网状态同步失败")
                            return False
                
                # 如果有任何更改，创建新会话
                if changes_made:
                    self.current_chat_id = None
                    if not self.create_chat():
                        logger.error("[Yuewen] 创建新会话失败")
                        return False
                        
                return True
                
            except Exception as e:
                if retry == 0:
                    # 如果是第一次失败,尝试刷新令牌后重试
                    if self.login_handler.refresh_token():
                        logger.info("[Yuewen] 令牌刷新成功,重试获取状态")
                        continue
                logger.error(f"[Yuewen] 同步服务器状态失败: {str(e)}")
                return False
        return False

    def _check_file_status(self, file_id):
        """检查文件状态"""
        max_retries = 5  # 最大重试次数
        retry_interval = 0.5  # 重试间隔(秒)
        
        headers = self._update_headers()
        headers.update({
            'Content-Type': 'application/json',
            'canary': 'false',
            'connect-protocol-version': '1',
            'oasis-appid': '10200',
            'oasis-mode': '2',
            'oasis-platform': 'web',
            'priority': 'u=1, i',
            'x-waf-client-type': 'fetch_sdk'
        })
        
        for i in range(max_retries):
            try:
                response = self.client.post(
                    'https://yuewen.cn/api/proto.file.v1.FileService/GetFileStatus',
                    headers=headers,
                    json={"id": file_id}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("fileStatus") == 1:  # 1表示成功
                        return True
                    elif not data.get("needFurtherCall", True):  # 如果不需要继续查询
                        return False
                elif response.status_code == 401:
                    if self.login_handler.refresh_token():
                        continue
                    return False
                    
                time.sleep(retry_interval)
            except Exception as e:
                logger.error(f"[Yuewen] 检查文件状态失败: {str(e)}")
                if i < max_retries - 1:  # 如果不是最后一次重试
                    time.sleep(retry_interval)
        
        return False

    def _process_multi_images(self, images, prompt, e_context):
        """处理多张图片"""
        try:
            if not self.current_chat_id:
                if not self.create_chat():
                    return "创建会话失败"
            
            # 构建带多图片的消息
            attachments = []
            for idx, img in enumerate(images, 1):
                attachments.append({
                    "attachmentType": "image/jpeg",
                    "attachmentId": img['file_id'],
                    "name": f"{idx}.jpg",
                    "width": str(img['width']),
                    "height": str(img['height']),
                    "size": str(img['size']),
                    "usedPercent": -1
                })
            
            message = {
                "chatId": self.current_chat_id,
                "messageInfo": {
                    "text": prompt,
                    "attachments": attachments,
                    "author": {"role": "user"}
                },
                "messageMode": "SEND_MESSAGE",
                "modelId": self.current_model_id
            }
            
            # 发送消息
            json_str = json.dumps(message, separators=(',', ':'), ensure_ascii=False)
            encoded = json_str.encode('utf-8')
            protocol_header = struct.pack('>BI', 0, len(encoded))
            
            headers = self._update_headers()
            headers['Content-Type'] = 'application/connect+json'
            
            start_time = time.time()
            response = self.client.post(
                'https://yuewen.cn/api/proto.chat.v1.ChatMessageService/SendMessageStream',
                headers=headers,
                content=protocol_header + encoded,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"[Yuewen] 发送多图片消息失败: HTTP {response.status_code}")
                return "请求失败，请重试"
            
            self.last_active_time = time.time()
            return self._parse_stream_response(response, start_time)
            
        except Exception as e:
            logger.error(f"[Yuewen] 处理多图片失败: {e}")
            return "处理失败，请重试"

    def _get_share_image(self, chat_id, messages):
        """获取分享图片"""
        try:
            # 第一步：获取分享ID
            headers = self._update_headers()
            headers.update({
                'Content-Type': 'application/json',
                'canary': 'false',
                'connect-protocol-version': '1',
                'oasis-appid': '10200',
                'oasis-mode': '2',
                'oasis-platform': 'web',
                'x-waf-client-type': 'fetch_sdk'
            })
            
            share_data = {
                "chatId": chat_id,
                "selectedMessageList": messages,
                "needTitle": True
            }
            
            response = self.client.post(
                'https://yuewen.cn/api/proto.chat.v1.ChatService/ChatShareSelectMessage',
                headers=headers,
                json=share_data
            )
            
            if response.status_code != 200:
                return None
                
            share_result = response.json()
            chat_share_id = share_result.get('chatShareId')
            if not chat_share_id:
                return None
                
            # 第二步：生成分享图片
            poster_data = {
                "chatShareId": chat_share_id,
                "pageSize": 10,
                "shareUrl": f"https://yuewen.cn/share/{chat_share_id}?utm_source=share&utm_content=web_image_share&version=2",
                "width": 430,
                "scale": 3
            }
            
            response = self.client.post(
                'https://yuewen.cn/api/proto.shareposter.v1.SharePosterService/GenerateChatSharePoster',
                headers=headers,
                json=poster_data
            )
            
            if response.status_code != 200:
                return None
                
            poster_result = response.json()
            return poster_result.get('staticUrl')
            
        except Exception as e:
            logger.error(f"[Yuewen] 获取分享图片失败: {e}")
            return None

    def _handle_video_ref_request(self, content, e_context, user_id):
        """处理参考图生成视频请求"""
        try:
            # 解析命令，检查是否需要润色
            prompt = content.replace("参考图", "", 1).strip()
            use_rephrase = False
            camera_list = []
            
            # 检查是否有润色标志
            if "-润色" in prompt:
                use_rephrase = True
                prompt = prompt.replace("-润色", "").strip()
            
            # 检查是否有镜头语言
            for short_name, full_name in self.camera_movements.items():
                if f"-{short_name}" in prompt:
                    camera_list.append(full_name)
                    prompt = prompt.replace(f"-{short_name}", "").strip()
            
            if not prompt:
                return "❌ 请提供提示词"
            
            # 检查是否有引用图片
            ref_type = e_context["context"].get("ref_type")
            ref_content = e_context["context"].get("ref_content")
            
            if ref_type == "image" and ref_content:
                logger.info("[Yuewen] 检测到引用图片，直接处理")
                try:
                    return self._handle_video_ref_image(ref_content, prompt, e_context, use_rephrase, camera_list)
                except Exception as e:
                    logger.error(f"[Yuewen] 处理引用图片失败: {e}")
                    return "❌ 处理引用图片失败，请重试"
            
            # 创建等待图片状态
            msg = e_context['context'].kwargs.get("msg")
            is_group = e_context['context'].kwargs.get("isgroup", False)
            
            # 生成等待ID
            if is_group:
                group_id = msg.other_user_id if msg else None
                real_user_id = msg.actual_user_id if msg and hasattr(msg, "actual_user_id") else None
                waiting_id = f"{group_id}_{real_user_id}" if real_user_id else group_id
            else:
                real_user_id = msg.from_user_id if msg else None
                waiting_id = real_user_id
            
            # 设置等待状态
            self.video_ref_waiting[waiting_id] = {
                'prompt': prompt,
                'use_rephrase': use_rephrase,
                'camera_list': camera_list
            }
            
            return "请发送一张参考图片用于生成视频"
            
        except Exception as e:
            logger.error(f"[Yuewen] 处理参考图视频请求失败: {e}")
            return "❌ 处理请求失败，请重试"

    def _handle_video_ref_image(self, image_path, prompt, e_context, use_rephrase=False, camera_list=None):
        """处理参考图生成视频"""
        try:
            # 开始计时
            start_time = time.time()
            
            # 获取图片数据
            msg = e_context['context'].kwargs.get('msg')
            logger.info("[Yuewen] 开始处理参考图生成视频请求")
            
            image_bytes = self._get_image_data(msg, image_path)
            if not image_bytes:
                return "❌ 获取图片失败，请重试"

            # 上传图片
            file_id = self._upload_image(image_bytes)
            if not file_id:
                return "❌ 图片上传失败，请重试"
            
            # 检查文件状态
            if not self._check_file_status(file_id):
                return "❌ 图片处理失败，请重试"
            
            # 润色提示词(如果需要)
            if use_rephrase:
                rephrased_prompt = self._rephrase_video_prompt(prompt)
                if rephrased_prompt:
                    logger.info(f"[Yuewen] 提示词润色成功: {prompt} -> {rephrased_prompt}")
                    prompt = rephrased_prompt
                else:
                    logger.warning("[Yuewen] 提示词润色失败，使用原始提示词")
                
            # 发送视频生成请求
            video_task = self._send_video_ref_request(prompt, file_id, camera_list)
            if not video_task:
                return "❌ 视频生成请求失败，请重试"
            
            video_sqid = video_task.get("videoSqid")
            if not video_sqid:
                return "❌ 无效的视频任务ID"
            
            # 发送正在处理的消息
            reply = Reply()
            reply.type = ReplyType.TEXT
            
            camera_info = f"使用镜头语言: {', '.join(camera_list)}" if camera_list else ""
            reply.content = f"视频生成任务已提交，预计需要3-5分钟，请耐心等待...\n提示词: {prompt}\n{camera_info}"
            e_context["channel"].send(reply, e_context["context"])
            
            # 轮询任务状态
            video_url = self._check_video_ref_status(video_sqid)
            if not video_url:
                return "❌ 视频生成失败或超时，请重试"
            
            # 计算总耗时
            total_time = time.time() - start_time
            logger.info(f"[Yuewen] 视频生成完成，耗时: {total_time:.2f}秒")
            
            # 发送成功消息
            success_message = f"✅ 视频生成成功！耗时: {total_time:.2f}秒"
            reply = Reply()
            reply.type = ReplyType.TEXT
            reply.content = success_message
            e_context["channel"].send(reply, e_context["context"])
            
            # 发送视频URL
            video_reply = Reply(ReplyType.VIDEO_URL, video_url)
            e_context["channel"].send(video_reply, e_context["context"])
            
            return None  # 返回None表示已经处理完毕
            
        except Exception as e:
            logger.error(f"[Yuewen] 处理参考图生成视频失败: {e}")
            return "❌ 处理失败，请重试"

    def _send_video_ref_request(self, prompt, file_id, camera_list=None):
        """发送参考图生成视频请求"""
        try:
            headers = self._update_headers()
            headers.update({
                'Content-Type': 'application/json',
                'canary': 'false',
                'connect-protocol-version': '1',
                'oasis-appid': '10200',
                'oasis-mode': '2',
                'oasis-platform': 'web',
                'x-waf-client-type': 'fetch_sdk'
            })
            
            image_url = f"https://yuewen.cn/api/storage?id={file_id}"
            
            # 构建请求数据
            data = {
                "prompt": prompt,
                "imageUrl": image_url,
                "type": "VIDEO_TASK_TYPE_IMAGE_TO_VIDEO"
            }
            
            # 添加镜头语言列表
            if camera_list and len(camera_list) > 0:
                data["cameraList"] = camera_list
            
            for retry in range(2):
                try:
                    response = self.client.post(
                        'https://yuewen.cn/api/proto.video.v1.VideoService/PostVideoTask',
                        headers=headers,
                        json=data
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "videoId" in data and "videoSqid" in data:
                            logger.info(f"[Yuewen] 视频任务创建成功: ID={data['videoId']}, SQID={data['videoSqid']}")
                            return data
                    elif response.status_code == 401 and retry == 0:
                        if self.login_handler.refresh_token():
                            continue
                        else:
                            logger.error("[Yuewen] Token刷新失败")
                    logger.error(f"[Yuewen] 创建视频任务失败: {response.text}")
                    return None
                except Exception as e:
                    if retry == 0:
                        continue
                    logger.error(f"[Yuewen] 创建视频任务失败: {str(e)}")
                    return None
            return None
        except Exception as e:
            logger.error(f"[Yuewen] 创建视频任务异常: {str(e)}")
            return None

    def _check_video_ref_status(self, task_id, max_retries=180):
        """检查视频生成状态"""
        retry_interval = 10  # 重试间隔为10秒
        last_status_time = 0  # 上次状态输出时间
        status_interval = 30  # 状态输出间隔(秒)
        
        headers = self._update_headers()
        headers.update({
            'Content-Type': 'application/json',
            'canary': 'false',
            'connect-protocol-version': '1',
            'oasis-appid': '10200',
            'oasis-mode': '2',
            'oasis-platform': 'web',
            'x-waf-client-type': 'fetch_sdk'
        })
        
        for i in range(max_retries):
            current_time = time.time()
            try:
                response = self.client.post(
                    'https://yuewen.cn/api/proto.video.v1.VideoService/GetVideoFeed',
                    headers=headers,
                    json={"videoSqid": task_id}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    feed = data.get("feed", {})
                    status = feed.get("status")
                    
                    # 输出任务状态(每30秒)
                    if current_time - last_status_time >= status_interval:
                        task_desc = feed.get("taskDesc", "处理中...")
                        logger.info(f"[Yuewen] 视频生成状态: {status}, {task_desc}")
                        last_status_time = current_time
                    
                    # 判断任务是否完成
                    if status == "GENERATED":
                        content = feed.get("content", {})
                        video_url = content.get("url")
                        if video_url:
                            logger.info(f"[Yuewen] 视频生成完成: {video_url}")
                            return video_url
                        else:
                            logger.error("[Yuewen] 视频地址为空")
                            return None
                    elif status == "FAILED":
                        logger.error(f"[Yuewen] 视频生成失败: {feed.get('failReason', '未知错误')}")
                        return None
                elif response.status_code == 401 and i < max_retries - 1:
                    if self.login_handler.refresh_token():
                        headers = self._update_headers()
                        headers.update({
                            'Content-Type': 'application/json',
                            'canary': 'false',
                            'connect-protocol-version': '1',
                            'oasis-appid': '10200',
                            'oasis-mode': '2',
                            'oasis-platform': 'web',
                            'x-waf-client-type': 'fetch_sdk'
                        })
                    else:
                        logger.error("[Yuewen] Token刷新失败")
                
                # 如果不是最后一次轮询，则等待
                if i < max_retries - 1:
                    time.sleep(retry_interval)
                
            except Exception as e:
                logger.error(f"[Yuewen] 检查视频状态失败: {str(e)}")
                if i < max_retries - 1:
                    time.sleep(retry_interval)
        
        logger.error(f"[Yuewen] 视频生成超时")
        return None

    def _refresh_token(self):
        """刷新令牌"""
        # 限制刷新频率，避免频繁请求
        current_time = time.time()
        if current_time - self.last_token_refresh < 60:  # 1分钟内只刷新一次
            logger.debug("[Yuewen] 令牌刷新频率限制，跳过")
            return False
        
        try:
            logger.info("[Yuewen] 开始刷新令牌")
            headers = {
                'accept': '*/*',
                'accept-language': 'zh-CN,zh;q=0.9',
                'canary': 'false',
                'connect-protocol-version': '1',
                'content-type': 'application/json',
                'cookie': f"Oasis-Webid={self.config['oasis_webid']}; Oasis-Token={self.config['oasis_token']}",
                'oasis-appid': '10200',
                'oasis-mode': '2',
                'oasis-platform': 'web',
                'oasis-webid': self.config['oasis_webid'],
                'origin': 'https://yuewen.cn',
                'priority': 'u=1, i',
                'x-waf-client-type': 'fetch_sdk'
            }
            
            response = self.client.post(
                'https://yuewen.cn/passport/proto.api.passport.v1.PassportService/RefreshToken',
                headers=headers,
                json={}
            )
            
            if response.status_code == 200:
                data = response.json()
                access_token = data.get('accessToken', {}).get('raw')
                refresh_token = data.get('refreshToken', {}).get('raw')
                
                if access_token and refresh_token:
                    # 更新令牌
                    self.config['oasis_token'] = f"{access_token}...{refresh_token}"
                    self.login_handler.config = self.config
                    self.login_handler.save_config()
                    
                    self.last_token_refresh = current_time
                    logger.info("[Yuewen] 令牌刷新成功")
                    return True
                
            logger.error(f"[Yuewen] 令牌刷新失败: HTTP {response.status_code}")
            return False
        except Exception as e:
            logger.error(f"[Yuewen] 令牌刷新异常: {str(e)}")
            return False

    def _handle_video_request(self, content, e_context):
        """处理文生视频请求"""
        try:
            # 解析命令
            # 处理润色和镜头语言选项
            prompt = content
            use_rephrase = False
            camera_list = []
            
            # 检查是否有润色标志
            if "-润色" in content:
                use_rephrase = True
                prompt = content.replace("-润色", "").strip()
            
            # 检查是否有镜头语言
            for short_name, full_name in self.camera_movements.items():
                if f"-{short_name}" in prompt:
                    camera_list.append(full_name)
                    prompt = prompt.replace(f"-{short_name}", "").strip()
            
            if not prompt:
                return "❌ 请提供视频生成提示词"
                
            # 润色提示词(如果需要)
            if use_rephrase:
                rephrased_prompt = self._rephrase_video_prompt(prompt)
                if rephrased_prompt:
                    logger.info(f"[Yuewen] 提示词润色成功: {prompt} -> {rephrased_prompt}")
                    prompt = rephrased_prompt
                else:
                    logger.warning("[Yuewen] 提示词润色失败，使用原始提示词")
                
            # 发送视频生成请求
            video_task = self._send_video_gen_request(prompt, camera_list=camera_list)
            if not video_task:
                return "❌ 视频生成请求失败，请重试"
                
            video_sqid = video_task.get("videoSqid")
            if not video_sqid:
                return "❌ 无效的视频任务ID"
                
            # 发送正在处理的消息
            reply = Reply()
            reply.type = ReplyType.TEXT
            
            camera_info = f"使用镜头语言: {', '.join(camera_list)}" if camera_list else ""
            reply.content = f"视频生成任务已提交，预计需要3-5分钟，请耐心等待...\n提示词: {prompt}\n{camera_info}"
            e_context["channel"].send(reply, e_context["context"])
            
            # 轮询任务状态
            video_url = self._check_video_status(video_sqid, max_retries=36)  # 最多等待6分钟
            if not video_url:
                return "❌ 视频生成失败或超时，请重试"
                
            # 发送成功消息
            success_message = f"✅ 视频生成成功！"
            reply = Reply()
            reply.type = ReplyType.TEXT
            reply.content = success_message
            e_context["channel"].send(reply, e_context["context"])
            
            # 发送视频URL
            video_reply = Reply(ReplyType.VIDEO_URL, video_url)
            e_context["channel"].send(video_reply, e_context["context"])
            
            return None  # 返回None表示已经处理完毕
            
        except Exception as e:
            logger.error(f"[Yuewen] 处理视频生成请求失败: {e}")
            return "❌ 处理请求失败，请重试"

    def _send_video_gen_request(self, prompt, resolution="992*544", use_pro_model=False, camera_list=None):
        """发送视频生成请求"""
        try:
            headers = self._update_headers()
            headers.update({
                'Content-Type': 'application/json',
                'canary': 'false',
                'connect-protocol-version': '1',
                'oasis-appid': '10200', 
                'oasis-mode': '2',
                'oasis-platform': 'web',
                'x-waf-client-type': 'fetch_sdk'
            })
            
            # 构建请求数据
            data = {
                "prompt": prompt,
                "type": "VIDEO_TASK_TYPE_TEXT_TO_VIDEO",
            }
            
            # 添加镜头语言列表
            if camera_list and len(camera_list) > 0:
                data["cameraList"] = camera_list
                
            for retry in range(2):
                try:
                    response = self.client.post(
                        'https://yuewen.cn/api/proto.video.v1.VideoService/PostVideoTask',
                        headers=headers,
                        json=data
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "videoId" in data and "videoSqid" in data:
                            logger.info(f"[Yuewen] 视频任务创建成功: ID={data['videoId']}, SQID={data['videoSqid']}")
                            return data
                    elif response.status_code == 401 and retry == 0:
                        if self._refresh_token():
                            headers = self._update_headers()
                            headers.update({
                                'Content-Type': 'application/json',
                                'canary': 'false',
                                'connect-protocol-version': '1',
                                'oasis-appid': '10200',
                                'oasis-mode': '2',
                                'oasis-platform': 'web',
                                'x-waf-client-type': 'fetch_sdk'
                            })
                            continue
                        else:
                            logger.error("[Yuewen] Token刷新失败")
                    logger.error(f"[Yuewen] 创建视频任务失败: {response.text}")
                    return None
                except Exception as e:
                    if retry == 0:
                        continue
                    logger.error(f"[Yuewen] 创建视频任务失败: {str(e)}")
                    return None
            return None
        except Exception as e:
            logger.error(f"[Yuewen] 创建视频任务异常: {str(e)}")
            return None

    def _check_video_status(self, task_id, max_retries=36):
        """检查视频生成状态"""
        retry_interval = 10  # 重试间隔为10秒
        last_status_time = 0  # 上次状态输出时间
        status_interval = 30  # 状态输出间隔(秒)
        
        headers = self._update_headers()
        headers.update({
            'Content-Type': 'application/json',
            'canary': 'false',
            'connect-protocol-version': '1',
            'oasis-appid': '10200',
            'oasis-mode': '2',
            'oasis-platform': 'web',
            'x-waf-client-type': 'fetch_sdk'
        })
        
        for i in range(max_retries):
            current_time = time.time()
            try:
                response = self.client.post(
                    'https://yuewen.cn/api/proto.video.v1.VideoService/GetVideoFeed',
                    headers=headers,
                    json={"videoSqid": task_id}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    feed = data.get("feed", {})
                    status = feed.get("status")
                    
                    # 输出任务状态(每30秒)
                    if current_time - last_status_time >= status_interval:
                        task_desc = feed.get("taskDesc", "处理中...")
                        logger.info(f"[Yuewen] 视频生成状态: {status}, {task_desc}")
                        last_status_time = current_time
                    
                    # 判断任务是否完成
                    if status == "GENERATED":
                        content = feed.get("content", {})
                        video_url = content.get("url")
                        if video_url:
                            logger.info(f"[Yuewen] 视频生成完成: {video_url}")
                            return video_url
                        else:
                            logger.error("[Yuewen] 视频地址为空")
                            return None
                    elif status == "FAILED":
                        logger.error(f"[Yuewen] 视频生成失败: {feed.get('failReason', '未知错误')}")
                        return None
                elif response.status_code == 401 and i < max_retries - 1:
                    if self._refresh_token():
                        headers = self._update_headers()
                        headers.update({
                            'Content-Type': 'application/json',
                            'canary': 'false',
                            'connect-protocol-version': '1',
                            'oasis-appid': '10200',
                            'oasis-mode': '2',
                            'oasis-platform': 'web',
                            'x-waf-client-type': 'fetch_sdk'
                        })
                    else:
                        logger.error("[Yuewen] Token刷新失败")
                
                # 如果不是最后一次轮询，则等待
                if i < max_retries - 1:
                    time.sleep(retry_interval)
                    
            except Exception as e:
                logger.error(f"[Yuewen] 检查视频状态失败: {str(e)}")
                if i < max_retries - 1:
                    time.sleep(retry_interval)
        
        logger.error(f"[Yuewen] 视频生成超时")
        return None

    def _rephrase_video_prompt(self, prompt):
        """润色视频提示词"""
        try:
            headers = self._update_headers()
            headers.update({
                'Content-Type': 'application/json',
                'canary': 'false',
                'connect-protocol-version': '1',
                'oasis-appid': '10200',
                'oasis-mode': '2',
                'oasis-platform': 'web',
                'x-waf-client-type': 'fetch_sdk'
            })
            
            for retry in range(2):
                try:
                    response = self.client.post(
                        'https://yuewen.cn/api/proto.video.v1.VideoService/RephraseVideoPrompt',
                        headers=headers,
                        json={"prompt": prompt}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "rephrasedPrompt" in data:
                            return data["rephrasedPrompt"]
                    elif response.status_code == 401 and retry == 0:
                        if self._refresh_token():
                            continue
                        else:
                            logger.error("[Yuewen] Token刷新失败")
                    logger.error(f"[Yuewen] 提示词润色失败: {response.text}")
                    return None
                except Exception as e:
                    if retry == 0:
                        continue
                    logger.error(f"[Yuewen] 提示词润色失败: {str(e)}")
                    return None
            return None
        except Exception as e:
            logger.error(f"[Yuewen] 提示词润色异常: {str(e)}")
            return None

    def _check_login_status(self):
        """检查登录状态"""
        # 检查配置是否完整
        if not self.config.get('oasis_webid') or not self.config.get('oasis_token'):
            logger.info("[Yuewen] 缺少登录信息，需要登录")
            return False
        
        # 尝试刷新token验证是否有效
        if self.login_handler.refresh_token():
            logger.info("[Yuewen] 登录状态有效")
            return True
        
        # 如果刷新失败，标记为需要登录
        self.config["need_login"] = True
        self.save_config(self.config)
        logger.info("[Yuewen] 登录已失效，需要重新登录")
        return False

    def _initiate_login(self, e_context, user_id):
        """发起登录流程"""
        try:
            # 如果没有注册设备，先注册
            if not self.config.get('oasis_webid'):
                if not self.login_handler.register_device():
                    reply = Reply()
                    reply.type = ReplyType.TEXT
                    reply.content = "❌ 设备注册失败，请稍后重试"
                    e_context['reply'] = reply
                    e_context.action = EventAction.BREAK_PASS
                    return
            
            # 提示用户输入手机号
            reply = Reply()
            reply.type = ReplyType.TEXT
            reply.content = "📱 请输入您的11位手机号码"
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS
            
            # 将用户标记为等待验证码状态(临时存储空字符串)
            self.waiting_for_verification[user_id] = ""
            return
        except Exception as e:
            logger.error(f"[Yuewen] 发起登录流程失败: {e}")
            reply = Reply()
            reply.type = ReplyType.TEXT
            reply.content = f"❌ 登录流程启动失败: {str(e)}"
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS
            return

    def _send_verification_code(self, e_context, user_id, phone_number):
        """发送验证码"""
        try:
            if self.login_handler.send_verify_code(phone_number):
                # 更新用户状态，保存手机号
                self.waiting_for_verification[user_id] = phone_number
                
                reply = Reply()
                reply.type = ReplyType.TEXT
                # 修改这里，移除手机号码显示
                reply.content = "✅ 验证码已发送，请输入收到的4位数验证码"
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS
                return
            else:
                # 发送验证码失败
                self.waiting_for_verification.pop(user_id, None)
                
                reply = Reply()
                reply.type = ReplyType.TEXT
                reply.content = "❌ 验证码发送失败，请重新发送\"yw登录\"进行登录"
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS
                return
        except Exception as e:
            logger.error(f"[Yuewen] 发送验证码失败: {e}")
            self.waiting_for_verification.pop(user_id, None)
            
            reply = Reply()
            reply.type = ReplyType.TEXT
            reply.content = f"❌ 验证码发送失败: {str(e)}"
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS
            return

    def _verify_login(self, e_context, user_id, verify_code):
        """验证登录"""
        try:
            # 获取之前保存的手机号
            phone_number = self.waiting_for_verification.get(user_id)
            if not phone_number:
                reply = Reply()
                reply.type = ReplyType.TEXT
                reply.content = "❌ 验证失败：请先发送手机号获取验证码"
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS
                return
            
            # 验证登录
            login_result = self.login_handler.verify_login(phone_number, verify_code)
            if login_result:
                # 清除等待状态
                self.waiting_for_verification.pop(user_id, None)
                
                # 确保配置保存成功 - 这里已经在 verify_login 方法内保存了
                
                # 创建新会话
                self.create_chat()
                
                reply = Reply()
                reply.type = ReplyType.TEXT
                reply.content = "✅ 登录成功！现在可以正常使用跃问功能了"
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS
                return
            else:
                reply = Reply()
                reply.type = ReplyType.TEXT
                reply.content = "❌ 验证码错误或已过期，请重新发送\"yw登录\"进行登录"
                e_context['reply'] = reply
                e_context.action = EventAction.BREAK_PASS
                return
        except Exception as e:
            logger.error(f"[Yuewen] 验证登录失败: {e}")
            self.waiting_for_verification.pop(user_id, None)
            
            reply = Reply()
            reply.type = ReplyType.TEXT
            reply.content = f"❌ 验证登录失败: {str(e)}"
            e_context['reply'] = reply
            e_context.action = EventAction.BREAK_PASS
            return

if __name__ == '__main__':
    client = YuewenPlugin()
    client.start_chat()