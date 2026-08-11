"""
快速测试BiomedCLIP集成
运行此脚本验证BiomedCLIP安装和基本功能
"""

import torch
import sys
import os

def test_biomedclip_installation():
    """测试BiomedCLIP是否正确安装"""
    print("="*60)
    print("📦 Step 1: Testing BiomedCLIP Installation")
    print("="*60)
    
    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
        print("✅ open_clip_torch installed successfully")
    except ImportError as e:
        print("❌ Failed to import open_clip")
        print(f"Error: {e}")
        print("\n💡 Solution: pip install open_clip_torch")
        return False
    
    try:
        print("\n🔄 Loading BiomedCLIP model...")
        model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        print("✅ BiomedCLIP model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load BiomedCLIP: {e}")
        return False
    
    return True


def test_feature_extraction():
    """测试图像特征提取"""
    print("\n" + "="*60)
    print("🖼️ Step 2: Testing Image Feature Extraction")
    print("="*60)
    
    try:
        from open_clip import create_model_from_pretrained
        from PIL import Image
        import numpy as np
        
        model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # 创建随机测试图像
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # 预处理
        image_tensor = preprocess(test_image).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
        
        print(f"✅ Image feature extraction successful")
        print(f"   Feature shape: {image_features.shape}")
        print(f"   Feature dimension: {image_features.shape[1]}")
        print(f"   Device: {device}")
        
        assert image_features.shape[1] == 512, "Expected 512-dim features"
        print("✅ Feature dimension validation passed (512-dim)")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_encoding():
    """测试文本编码"""
    print("\n" + "="*60)
    print("📝 Step 3: Testing Text Encoding")
    print("="*60)
    
    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
        
        model, _ = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # 测试医学文本
        test_texts = [
            "adenocarcinoma histopathology",
            "normal tissue histopathology"
        ]
        
        # 使用正确的tokenizer(避免CUDA索引错误)
        text_tokens = tokenizer(test_texts).to(device)
        
        # 编码
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
        
        print(f"✅ Text encoding successful")
        print(f"   Input texts: {len(test_texts)}")
        print(f"   Text feature shape: {text_features.shape}")
        print(f"   Texts: {test_texts}")
        
        assert text_features.shape == (2, 512), "Expected [2, 512] shape"
        print("✅ Text feature dimension validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Text encoding failed: {e}")
        # CUDA错误需要重置设备
        if 'CUDA' in str(e):
            print("💡 Resetting CUDA device...")
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        import traceback
        traceback.print_exc()
        return False


def test_model_import():
    """测试自定义模型导入"""
    print("\n" + "="*60)
    print("🧪 Step 4: Testing Custom Model Import")
    print("="*60)
    
    try:
        # nmslib是可选依赖,仅用于HIPT模型,可跳过
        try:
            import nmslib
            print("✅ nmslib found (optional dependency)")
        except ImportError:
            print("⚠️  nmslib not found (optional, only needed for HIPT - skipping)")
        
        # 添加项目路径
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
        print("✅ ViLa_MIL_BiomedCLIP imported successfully")
        
        # 创建测试配置
        class TestConfig:
            input_size = 512
            hidden_size = 256
            text_prompt = ["Adenocarcinoma", "NonAdenocarcinoma"]
            prototype_number = 16
        
        # 初始化模型
        model = ViLa_MIL_BiomedCLIP(
            config=TestConfig(),
            num_classes=2
        )
        print("✅ Model initialized successfully")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction_utils():
    """测试特征提取工具函数"""
    print("\n" + "="*60)
    print("🔧 Step 5: Testing Feature Extraction Utils")
    print("="*60)
    
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(project_root, 'feature_extraction'))
        
        from feature_extraction.patch_extraction_utils_biomedclip import (
            get_biomedclip_transforms,
            extract_text_features_biomedclip
        )
        print("✅ Utils imported successfully")
        
        # 测试transforms
        transforms = get_biomedclip_transforms()
        print(f"✅ Transforms created: {type(transforms)}")
        
        # 测试文本特征提取
        test_prompts = [
            "adenocarcinoma histopathology",
            "normal tissue histopathology"
        ]
        
        text_features = extract_text_features_biomedclip(
            test_prompts,
            output_path='test_text_features.pt'
        )
        
        print(f"✅ Text features extracted: {text_features.shape}")
        
        # 清理测试文件
        if os.path.exists('test_text_features.pt'):
            os.remove('test_text_features.pt')
            print("✅ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("# BiomedCLIP Integration Test Suite")
    print("#"*60 + "\n")
    
    tests = [
        ("Installation", test_biomedclip_installation),
        ("Feature Extraction", test_feature_extraction),
        ("Text Encoding", test_text_encoding),
        ("Model Import", test_model_import),
        ("Utils Functions", test_feature_extraction_utils),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # 总结
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print("\n" + "="*60)
    print(f"Results: {total_passed}/{total_tests} tests passed")
    print("="*60)
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! BiomedCLIP integration is ready.")
        print("\n📖 Next Steps:")
        print("   1. Extract features: python feature_extraction/patch_extraction_biomedclip.py")
        print("   2. Train model: python main.py --model_type ViLa_MIL_BiomedCLIP")
        print("   3. See BIOMEDCLIP_INTEGRATION_GUIDE.md for details")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        print("💡 Common solutions:")
        print("   - Install dependencies: pip install open_clip_torch")
        print("   - Check internet connection for model download")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
