from dataclasses import field
from gettext import translation
from fairseq.models.transformer import TransformerModel
from flask import request, jsonify, Flask
import logging
import json

from flask.templating import render_template
from interaction_helper import InteractionHelper
from mosestokenizer import *

from flask.wrappers import Response

from translation_connector import TranslationConnector

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
PORT = TranslationConnector.PORT
HOST = TranslationConnector.HOST

trained_model = dict()
logger = logging.getLogger("main")


@app.route('/', methods=['GET', 'POST'])
def translate_home():
    if request.method == "GET":
        return render_template('greeting.html', question='Hello world!', translation_ids=list(trained_model))
    question = request.form['question']
    translation_id = request.form['translation_id']
    logger.info(question)
    logger.info(translation_id)

    translation_info = dec_model_info(translation_id)
    src_sent = question
    src_lang = translation_info.get("srclang")
    tgt_lang = translation_info.get("tgtlang")
    template = translation_info.get("template")
    model_id = translation_info.get("modelid", "transformer")
    out = __translate(src_sent, src_lang, tgt_lang, model_id, template=template)
    
    return render_template('greeting.html', question=question, answer=out.get("result"), 
        detail_ans=json.dumps(out, indent=2, ensure_ascii=False), translation_ids=list(trained_model), translation_id_cur=translation_id)


def jsonstr_return(data):
    json_response = json.dumps(data, ensure_ascii = False)
    #creating a Response object to set the content type and the encoding
    response = Response(json_response, content_type="application/json; charset=utf-8" )
    return response

@app.route('/list-models', methods=['GET'])
def list_models():
    out = {'result': list(trained_model)}
    logger.info(out)
    return jsonify(out)


def __translate(src_sent, src_lang, tgt_lang, model_id, template=None):

    try:
        translation_info = enc_model_info(src_lang=src_lang, tgt_lang=tgt_lang, model_id=model_id, template=template)
        if translation_info in trained_model:
            try:
                tokenize = MosesTokenizer(src_lang)
                src_sent = " ".join(tokenize(src_sent))
            except Exception as e:
                pass

            tgt_sent = trained_model[translation_info].translate(src_sent)
            tgt_sent = tgt_sent.replace("@@ ", "")
            optional_info = None
            if isinstance(tgt_sent, tuple):
                tgt_sent, optional_info = tgt_sent[0], tgt_sent[1]
            out = {'result': tgt_sent,
                   'optional_info': optional_info,
                   }
        else:
            out = {'result': None,
                   'optional_info': 'model "{}" is not found'.format(translation_info),
                   }
    except Exception as e:
        print(e)
        out = {'result': None,
               'optional_info': str(e),
               }
    out['translation_info'] = translation_info
    logger.info(out)
    return out


@app.route('/translate-debug', methods=['GET', 'POST'])
def translate_debug():
    src_sent = request.args.get("srcsent")
    src_lang = request.args.get("srclang")
    tgt_lang = request.args.get("tgtlang")
    template = request.args.get("template")
    model_id = request.args.get("modelid", "transformer")
    out = __translate(src_sent, src_lang, tgt_lang, model_id, template)
    return jsonstr_return(out)


@app.route('/translate', methods=['POST'])
def translate():
    inputs = request.get_json(force=True)
    logger.info(inputs)
    src_sent = inputs.get("srcsent")
    src_lang = inputs.get("srclang")
    tgt_lang = inputs.get("tgtlang")
    template = inputs.get("template")
    translation_model_id = inputs.get("modelid", "transformer")

    out = __translate(src_sent, src_lang, tgt_lang, translation_model_id, template)

    return jsonstr_return(out)


def enc_model_info(**kwargs):
    id_str = ""
    for k in sorted(kwargs):
        v = kwargs[k]
        id_str = id_str + (":{}={}" if len(id_str) > 0 else "{}={}").format(k.replace("_", ""),v) 
    return id_str

def dec_model_info(id_str):
    fields = id_str.split(":")
    info = {}
    for f in fields:
        ff = f.split("=")
        if len(ff) > 1:
            info[ff[0]] = ff[1]
    return info

def __reset_all_models(path_config_file='modelinfo.json'):
    global trained_model 
    trained_model = {}
    models_info = json.load(open(path_config_file, encoding="utf8"))
    for m in models_info:
        src_lang = m['translation_info']['srclang']
        tgt_lang = m['translation_info']['tgtlang']
        model_id = m['translation_info']['modelid']
        template = m['translation_info'].get('template')
        translationid = enc_model_info(src_lang=src_lang, tgt_lang=tgt_lang, model_id=model_id, template=template)
        print("Loading model [{}] ...".format(translationid))
        try:
            if model_id == "transformer":
                trained_model[translationid] = TransformerModel.from_pretrained(
                    **m['model_info']
                ).cuda()
            else:
                # template translation 
                data_path = m['model_info']['data_path']
                m['model_info'].pop('data_path')

                input_args = [data_path]
                for k, v in m['model_info'].items():
                    input_args.append("--{}".format(k))
                    input_args.append("{}".format(v))

                translation_helper = InteractionHelper(input_args=input_args)
                trained_model[translationid] = translation_helper

        except Exception as e:
            import traceback
            traceback.print_stack()
            print(e)
            print("ERR when loading model: {}".format(m))
    out = {'result': list(trained_model)}
    return out


@app.route('/reload-all-model', methods=['GET'])
def reset_all_model():
    return jsonify(__reset_all_models())


"""
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"srclang":"de","tgtlang":"en", "srcsent": "wir werden alle geboren . wir bringen kinder zur welt ."}' \
  http://150.65.242.92:8008/translate 


curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"srclang":"de","tgtlang":"en_pred_probt", "srcsent": "wir werden alle geboren . wir bringen kinder zur welt ."}' \
  http://150.65.180.43:8008/translate 


curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"srclang":"de","tgtlang":"en", \
      "template": "en_pred_probt", \
      "modelid": "template-transformer", \
       "srcsent": "und der entscheidende indikator dafür ist das aussterben der sprachen ."}' \
  http://localhost:8008/translate 

===============
en-vi

 curl --header "Content-Type: application/json" \
 --request POST \
 --data '{"srclang":"en","tgtlang":"vi","template": "vi_pred_probt", "modelid": "template-transformer", "srcsent": "this is a test ."}' \
 http://150.65.242.92:8008/translate

 curl --header "Content-Type: application/json" \
 --request POST \
 --data '{"srclang":"en","tgtlang":"vi", "modelid": "transformer", "srcsent": "this is a test ."}' \
 http://150.65.242.92:8008/translate

"""

if __name__ == "__main__":
    print(__reset_all_models(path_config_file='src/service/modelinfo.json'))
    print(__translate('wir werden alle geboren . wir bringen kinder zur welt .', 'de', 'en', 'transformer'))
    app.run(host=HOST, port=PORT)
