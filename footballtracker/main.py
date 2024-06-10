import argparse

from footballtracker import FootballTracker, ConfigManager


def main():
    parser = argparse.ArgumentParser(description='FootyTracker - Track players and the ball in a football video.')
    parser.add_argument('-i', '--input', required=True, help='Path to input video file.')
    parser.add_argument('-c', '--config', required=True, help='Path to configuration file.')
    parser.add_argument('-tpu', '--tpu',
                        action='store_true',
                        default=False,
                        help='Activate TPU acceleration (TPU must be connected).'
                        )
    parser.add_argument('-s', '--show_live', action='store_true', help='Show live detections.')

    args = parser.parse_args()

    cfg_manager = ConfigManager(args.config)
    cfg_manager.set('input_video_path', args.input)
    cfg_manager.set('show_live', args.show_live)
    cfg_manager.set('TPU_optimization', args.tpu)

    footy_tracker = FootballTracker(cfg_manager)
    footy_tracker.run()


if __name__ == '__main__':
    main()
