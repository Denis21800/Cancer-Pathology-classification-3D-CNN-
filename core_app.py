from pipeline import PipelineProcessor


config_path = 'config/config_m.json'

if __name__ == '__main__':
    pipeline = PipelineProcessor(config_path=config_path)
    pipeline.process_pipeline()
