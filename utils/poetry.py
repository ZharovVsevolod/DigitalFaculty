import heapq
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def ensure_length(txt, out_len, pad_value):
    if len(txt) < out_len:
        txt = list(txt) + [pad_value] * (out_len - len(txt))
    else:
        txt = txt[:out_len]
    return txt

def load_chunks(fname, chunk_size=200, more_space=True):
    # Открываем файл, читаем и записываем весь текст в массив по символам
    with open(fname, 'r', encoding="utf-8") as fin:
        full_text = fin.read()
    
    # Делим массив по символам на предложения длиной в chunk_size
    full_text = [full_text[start:start + chunk_size] for start in range(0, len(full_text), chunk_size // 2)]
    # Удаляем /n и заменяем их на пробел
    if more_space:
        full_text = [line.replace(u"\n", u" ") for line in full_text]
    
    return full_text

class LanguageModelDataset(Dataset):
    """
    Dataset для языковой модели\n
    Возвращает предложение в двух экземплярах:
        - предложение без последнего токена (seed_part)
        - предложение без начального токена (target_part)
    Делит предложение на две части: "начало" и "конец"\n
    При созданиии LanguageModelDataset:
    :param token_ids: - токены предложений
    :param chunk_length: - длина предложения (для того чтобы обрезать или дополнить паддингом)
    :param pad_value: - значение паддинга, если предложение окажется короче, чем нужно\n
    При вызове LanguageModelDataset:
    :param index: - номер предложения
    :return: кортеж из двух элементов:
        - преложение токенов, что модель получит в качестве входа (seed_part)
        - предложение токенов, что модель будет пытаться получить (target_part)
    """
    def __init__(self, token_ids, chunk_length=100, pad_value=0):
        self.token_ids = token_ids
        self.chunk_length = chunk_length
        self.pad_value = pad_value

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, item):
        text = self.token_ids[item]
        # Режем предложение в случайном месте по размеру чанка - обеспечим некоторую аугментацию данных
        start_i = random.randint(0, max(0, len(text) - self.chunk_length - 1))
        chunk = text[start_i : start_i + self.chunk_length + 1]
        
        # seed_part - то, что нейросеть будет видеть
        seed_part = chunk[:-1]
        # target_part - то, что нейросеть должна предсказать
        target_part = chunk[1:]

        # Дополняем паддингом предложение в случае, если оно короче, чем размер чанка
        seed_part = ensure_length(seed_part, self.chunk_length, self.pad_value)
        target_part = ensure_length(target_part, self.chunk_length, self.pad_value)

        seed_part = np.array(seed_part)
        target_part = np.array(target_part)

        return seed_part, target_part


# class LanguageModelDataset_for_Discriminator(Dataset):
#     def __init__(self, token_ids, target_icons, chunk_length=100, pad_value=0):
#         self.token_ids = token_ids
#         self.target_icons = target_icons
#         self.chunk_length = chunk_length
#         self.pad_value = pad_value
    
#     def __len__(self):
#         return len(self.token_ids)

#     def __getitem__(self, index):
#         text = self.token_ids[index]
#         start_i = random.randint(0, max(0, len(text) - self.chunk_length - 1))
#         chunk = text[start_i : start_i + self.chunk_length + 1]

#         seed_part = ensure_length(chunk, self.chunk_length, self.pad_value)
#         seed_part = np.array(seed_part)

#         return seed_part, self.target_icons[index]

class Dataset_for_GAN(Dataset):
    """
    Dataset для генеративно-состоязательной сети\n
    Делит предложение на две части: "начало" и "конец"\n
    При созданиии Dataset_for_GAN:
    :param token_ids: - токены предложений
    :param start_len: - длина начала предложения - тот текст, которая послужит началом для сгенерированного предложения
    :param end_len: - длина конца предложения - тот текст, который будет служить "эталонным" продолжением фразы
    :param pad_value: - значение паддинга, если предложение окажется короче, чем нужно\n
    При вызове Dataset_for_GAN:
    :param index: - номер предложения
    :return: кортеж из двух элементов:
        - "начало" предложения
        - "конец" предложения
    """
    def __init__(self, token_ids, start_len=25, end_len=50, pad_value=0):
        self.token_ids = token_ids
        self.start_len=start_len
        self.end_len = end_len
        self.pad_value = pad_value

    def __len__(self):
        return len(self.token_ids)
    
    def __getitem__(self, index):
        text = self.token_ids[index]
        # Обрезаем предложение, делаем из него две части - "начало" предложения и "конец"
        text_start = text[1:self.start_len+1]
        text_end = text[self.start_len+1:self.start_len + self.end_len + 1]
        # Сверяемся с необходимой длиной, в случае несовпадения дополняем паддингом
        text_end = ensure_length(text_end, self.end_len, self.pad_value)

        return np.array(text_start), np.array(text_end)

class GreedyGenerator:
    def __init__(self, model, tokenizer, device='cuda', eos_token_id=3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.eos_token_id = eos_token_id

    def __call__(self, seed_text, max_steps_n=40):
        seed_tokens = self.tokenizer.encode([seed_text])[0]

        for _ in range(max_steps_n):
            in_batch = torch.tensor(seed_tokens).unsqueeze(0).to(self.device)
            best_next_token = self.model(in_batch)[0, -1].argmax()
            if best_next_token == self.eos_token_id:
                break

            seed_tokens.append(best_next_token)

        return self.tokenizer.decode([seed_tokens])[0]


class GreedyGenerator_GAN:
    """
    Жадная генерация для генеративно-состязательной сети\n
    Модель генерирует продолжение предложения токен за токеном, пока не наберёт необходимую длину, после чего на выход подаётся её ответ\n
    При созданиии GreedyGenerator_GAN:
    :param model: - Модель для генерации
    :param device: cuda/cpu - где будут производиться вычисления (по умолчанию cuda)\n
    При вызове GreedyGenerator_GAN:
    :param seed_text: - "Начала" предложений, по которым будет производится генерация
    :param max_steps_n: - длина сгенерированных предложений
    :return: сгенерированные продолжения предложений
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def __call__(self, seed_text, max_steps_n=40):
        for greed_step in range(max_steps_n):
            next_token = self.model(seed_text)
            new = next_token[:, -1].argmax(dim=1)
            new = new.unsqueeze(0).transpose(0, 1)            
            
            # Для продолжения генерации изменяем seed_text
            seed_text = torch.cat((seed_text, new), 1)

            # В ответ и на выход пойдёт только сгенерированная часть
            if greed_step == 0:
                answer = new
            else:
                answer = torch.cat((answer, new), 1)

        return answer

def reweight(original, device, temperature=0.5, alpha=0):
    """
    Функция для перевзвешивания весов по формуле с двумя параметрами:
        - температура
        - альфа\n
    Понижает "уверенность" модели в предсказанном токене для улучшения генерации
    :param original: - изначальное распределение весов
    :param device: - устройство, на котором проводятся вычисления
    :param temperature: - параметр температура
    :param alpha: - параметр альфа
    :return: новое распределение весов
    """
    # Перевод массива весов в numpy
    original = original.cpu().detach().numpy()

    # Если есть параметр альфа, его применяем по формуле
    if alpha != 0:
        original = (1 - alpha) * original + alpha / len(original)
    # Делим логарифм весов на температуру для усреднения весов, сила которого зависит от температуры
    distribution = np.log(original) / temperature

    # Перевод перевзвешенного массива весов обратно в тензор
    distribution = torch.tensor(distribution).to(device)
    return distribution

class BeamGenerator:
    """
    Поисково-лучевая генерация продолжения предложения\n
    При созданиии BeamGenerator:
    :param model: - Модель для генерации
    :param tokenizer: - токенизатор для перевода текста с токенов в символы и обратно
    :param device: cuda/cpu - где будут производиться вычисления (по умолчанию cuda)
    :param eos_token_id: - номер токена, означающий конец предложения (End Of String, <EOS>)\n
    При вызове BeamGenerator:
    :param seed_text: - "Начало" предложения, по которому будет производится генерация
    :param max_steps_n: - длина сгенерированных предложений
    :param return_hypotheses_n: - сколько возвращает "гипотез" продолжения предложений (по умолчанию = 5)
    :param beamsize: - ширина луча (по умолчанию = 5)
    :param temperature: float (от 0.0 до 1.0) - температура генерации, который уменьшает "уверенность" модели в выборе следующего токена (по умолчанию = 0.5)
    :param alpha: float (от 0.0 до 1.0) - параметр для перевзвешивания для уменьшения "уверенности" модели в выборе следующего токена (по умолчанию = 0)
    :param need_reweight: - ключ, будет ли перевзвешивание весов ответов для уменьшения "уверенности" модели (по умолчанию False)
    :param without_score: - ключ, нужно ли возвращать дополнительно общий вес сгенерированного продолжения предложения (по умолчанию = False)
    :param need_to_encode: - ключ, нужно ли переводить seed_text в токены (по умолчанию = True. Изменить на False, если предложение подаётся сразу в токенах)
    :return: 
        - если without_score = False, то сгенерированное продолжение предложения
        - если without_score = True, то кортеж из двух элементов: 
            - вес предложения
            - сгенерированное продолжение предложения
        
    """
    def __init__(self, model, tokenizer, device='cuda', eos_token_id=3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.eos_token_id = eos_token_id

    def __call__(self, seed_text, max_steps_n=40, return_hypotheses_n=5, beamsize=5, temperature=0.5, alpha=0, need_reweight=False, without_score=False, need_to_encode=True):
        # При необходимости переводим предложение из символов в токены
        if need_to_encode:
            seed_tokens = self.tokenizer.encode([seed_text])[0]
        else:
            seed_tokens = seed_text
        initial_length = len(seed_tokens)

        partial_hypotheses = [(0, seed_tokens)]
        final_hypotheses = []

        while len(partial_hypotheses) > 0:
            # Создаём очередь для весов и токенов
            cur_partial_score, cur_partial_hypothesis = heapq.heappop(partial_hypotheses)
            
            # Генерируем первый токен
            in_batch = torch.tensor(cur_partial_hypothesis).unsqueeze(0).to(self.device)
            next_tokens_logits = self.model(in_batch)[0, -1]

            # При необходимости перевзвешиваем веса
            if need_reweight:
                next_tokens_logproba = reweight(next_tokens_logits, self.device, temperature, alpha)
            
            # Выбираем топ-beamsize лучших вариантов токенов по весам
            next_tokens_logproba = F.log_softmax(next_tokens_logits)
            topk_continuations = next_tokens_logproba.topk(beamsize)

            for token_score, token_idx in zip(topk_continuations.values, topk_continuations.indices):
                token_score = float(token_score)
                token_idx = int(token_idx)

                # Считаем новый score для топ-beamsize вариантов
                old_denorm_score = cur_partial_score * np.sqrt(len(cur_partial_hypothesis))
                new_score = (old_denorm_score - token_score) / np.sqrt(len(cur_partial_hypothesis) + 1)

                # Берём новый токен и добавляем в конец возможного предложения
                new_hypothesis = cur_partial_hypothesis + [token_idx]
                # Создаётся новая единица - предложение с новым токеном и его вес
                new_item = (new_score, new_hypothesis)

                # Если токен конца предложения или досточная длина - записываем в финальный вариант
                if token_idx == self.eos_token_id or len(new_hypothesis) - initial_length >= max_steps_n:
                    final_hypotheses.append(new_item)
                else:
                    # Иначе добавляем в очередь
                    heapq.heappush(partial_hypotheses, new_item)

            # Если нагенерили достаточно по ширине луча поиска, лучшие (меньшие) топ-beamsize берём
            if len(partial_hypotheses) > beamsize:
                partial_hypotheses = heapq.nsmallest(beamsize, partial_hypotheses)
                heapq.heapify(partial_hypotheses)

        final_scores, final_token_lists = zip(*final_hypotheses)
        final_texts = self.tokenizer.decode(list(final_token_lists))

        # if without_score:
        #     return final_texts[:return_hypotheses_n][0]
        # else:
        #     result = list(zip(final_scores, final_texts))
        #     result.sort()
        #     result = result[:return_hypotheses_n]

        #     return result

        result = list(zip(final_scores, final_texts))
        result.sort()
        result = result[:return_hypotheses_n]

        if without_score:
            final_scores, result = zip(*result)
        
        return result
        
class Gen_method():
    """
    Класс для генерации нескольких разных продолжений предложения\n
    При созданиии Gen_method:
    :param gen_model: - Модель для генерации
    :param tokenizer: - токенизатор для перевода текста с токенов в символы и обратно
    При вызове Gen_method:
    :param text: - Начало предложения
    :param N_cut: - длина сгенерированных предложений
    :param is_string: - ключ, является ли text набором символов (string) или токенов (list). (По умолчанию - True, если передаётся в text массив токенов, установить на False)
    :return: сгенерированные продолжения предложений
    """
    def __init__(self, gen_model, tokenizer):
        self.tokenizer = tokenizer
        self.beam = BeamGenerator(gen_model, self.tokenizer)
    
    def __call__(self, text, N_cut = 20, is_string = True, need_reweight=False, temper=0.3, alpha=0):
        x = text[:N_cut]
        if not is_string:
            x = self.tokenizer.decode(x)[0]
        output = self.beam(x, return_hypotheses_n=1, without_score=True, need_reweight=need_reweight, temperature=temper, alpha=alpha)
        return output